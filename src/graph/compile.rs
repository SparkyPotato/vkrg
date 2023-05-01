use std::{collections::BTreeMap, hash::BuildHasherDefault, hint::unreachable_unchecked, iter::Peekable, ptr::NonNull};

use ash::vk::{
	AccessFlags2,
	BufferUsageFlags,
	Format,
	Image,
	ImageCreateFlags,
	ImageLayout,
	ImageMemoryBarrier2,
	ImageSubresourceRange,
	ImageUsageFlags,
	PipelineStageFlags2,
	REMAINING_ARRAY_LAYERS,
	REMAINING_MIP_LEVELS,
};
use tracing::{span, Level};

use crate::{
	arena::{Arena, IteratorAlloc},
	device::Device,
	graph::{
		virtual_resource::{
			compatible_formats,
			BufferUsage,
			ImageUsageType,
			ResourceLifetime,
			VirtualResourceData,
			VirtualResourceType,
		},
		ArenaMap,
		Frame,
		GpuBufferHandle,
		ImageUsage,
		PassData,
		RenderGraph,
		UploadBufferHandle,
	},
	resource::{image_aspect_mask, sampled_image_layout, BufferDesc, ImageDesc},
	Result,
};

pub(super) struct CompiledFrame<'pass, 'graph> {
	pub passes: Vec<PassData<'pass, 'graph>, &'graph Arena>,
	pub sync: Vec<Synchronization<'graph>, &'graph Arena>,
	pub resource_map: ResourceMap<'graph>,
	pub graph: &'graph mut RenderGraph,
}

pub struct Synchronization<'graph> {
	pub barriers: ArenaMap<'graph, Access, Access>,
	pub image_barriers: Vec<ImageMemoryBarrier2, &'graph Arena>,
}

#[derive(Copy, Clone)]
pub enum DataState {
	Uninit,
	Init { drop: fn(NonNull<()>) },
}

#[derive(Copy, Clone)]
pub struct Usage<U> {
	pub write: bool,
	pub usage: U,
}

pub struct GpuResource<'graph, H, U> {
	pub handle: H,
	pub usages: BTreeMap<u32, Usage<U>, &'graph Arena>,
}

pub enum Resource<'graph> {
	Data(NonNull<()>, DataState),
	UploadBuffer(UploadBufferHandle),
	GpuBuffer(GpuResource<'graph, GpuBufferHandle, BufferUsage>),
	Image(GpuResource<'graph, Image, ImageUsage>),
}

impl<'graph> Resource<'graph> {
	pub unsafe fn data<T>(&mut self) -> (NonNull<T>, &mut DataState) {
		match self {
			Resource::Data(ptr, state) => (ptr.cast(), state),
			_ => unreachable_unchecked(),
		}
	}

	pub unsafe fn upload_buffer(&self) -> UploadBufferHandle {
		match self {
			Resource::UploadBuffer(h) => *h,
			_ => unreachable_unchecked(),
		}
	}

	pub unsafe fn gpu_buffer(&self) -> &GpuResource<'graph, GpuBufferHandle, BufferUsage> {
		match self {
			Resource::GpuBuffer(res) => res,
			_ => unreachable_unchecked(),
		}
	}

	pub unsafe fn image(&self) -> &GpuResource<'graph, Image, ImageUsage> {
		match self {
			Resource::Image(res) => res,
			_ => unreachable_unchecked(),
		}
	}
}

pub struct ResourceMap<'graph> {
	resource_map: Vec<u32, &'graph Arena>,
	resources: Vec<Resource<'graph>, &'graph Arena>,
	buffers: Vec<u32, &'graph Arena>,
	images: Vec<u32, &'graph Arena>,
}

impl<'graph> ResourceMap<'graph> {
	unsafe fn new(
		resource_map: Vec<u32, &'graph Arena>, resources: Vec<Resource<'graph>, &'graph Arena>,
		buffers: Vec<u32, &'graph Arena>, images: Vec<u32, &'graph Arena>,
	) -> Self {
		Self {
			resource_map,
			resources,
			buffers,
			images,
		}
	}

	fn map_res(&self, res: u32) -> u32 {
		*self
			.resource_map
			.get(res as usize)
			.expect("resource ID from previous frame used")
	}

	fn arena(&self) -> &'graph Arena { self.resources.allocator() }

	fn buffers(&self) -> impl Iterator<Item = &GpuResource<'graph, GpuBufferHandle, BufferUsage>> {
		self.buffers.iter().map(move |&id| unsafe {
			let res = self.resources.get_unchecked(id as usize);
			res.gpu_buffer()
		})
	}

	fn images(&self) -> impl Iterator<Item = &GpuResource<'graph, Image, ImageUsage>> {
		self.images.iter().map(move |&id| unsafe {
			let res = self.resources.get_unchecked(id as usize);
			res.image()
		})
	}

	pub fn cleanup(self) -> usize {
		for resource in self.resources {
			match resource {
				Resource::Data(ptr, state) => {
					if let DataState::Init { drop } = state {
						drop(ptr);
					}
					unsafe { self.resource_map.allocator().deallocate(ptr.cast()) }
				},
				// Handled by cache reset.
				Resource::Image { .. } | Resource::GpuBuffer { .. } | Resource::UploadBuffer(_) => {},
			}
		}
		self.resource_map.len()
	}

	pub fn get(&mut self, res: u32) -> &'_ mut Resource<'graph> {
		let i = self.map_res(res) as usize;
		unsafe { self.resources.get_unchecked_mut(i) }
	}
}

#[derive(Eq, PartialEq, Hash)]
enum MergeCandidate {
	GpuBuffer,
	Image(super::ImageDesc),
}

enum ResourceDescType<'graph> {
	Data(NonNull<()>),
	UploadBuffer(BufferDesc),
	GpuBuffer(GpuResource<'graph, BufferDesc, BufferUsage>),
	Image(GpuResource<'graph, ImageDesc, ImageUsage>),
}

impl<'graph> VirtualResourceType<'graph> {
	fn to_res(self, pass: u32) -> ResourceDescType<'graph> {
		match self {
			VirtualResourceType::Data(ptr) => ResourceDescType::Data(ptr),
			VirtualResourceType::UploadBuffer(desc) => ResourceDescType::UploadBuffer(BufferDesc {
				size: desc.desc,
				usage: desc
					.read_usages
					.iter()
					.map(|(_, &x)| x.into())
					.fold(desc.write_usage.into(), |a: BufferUsageFlags, b: BufferUsageFlags| {
						a | b
					}),
			}),
			VirtualResourceType::GpuBuffer(desc) => ResourceDescType::GpuBuffer(GpuResource {
				handle: BufferDesc {
					size: desc.desc,
					usage: desc
						.read_usages
						.iter()
						.map(|(_, &x)| x.into())
						.fold(desc.write_usage.into(), |a: BufferUsageFlags, b: BufferUsageFlags| {
							a | b
						}),
				},
				usages: {
					let arena = *desc.read_usages.allocator();
					desc.read_usages
						.into_iter()
						.map(|(pass, usage)| (pass, Usage { write: false, usage }))
						.chain(std::iter::once((
							pass,
							Usage {
								write: true,
								usage: desc.write_usage,
							},
						)))
						.collect_in(arena)
				},
			}),
			VirtualResourceType::Image(desc) => {
				let mut usages = BTreeMap::new_in(*desc.read_usages.allocator());

				let mut usage: ImageUsageFlags = desc.write_usage.usage.into();
				let mut flags = desc.write_usage.create_flags();
				usages.insert(
					pass,
					Usage {
						write: true,
						usage: desc.write_usage,
					},
				);

				for (&pass, &read) in desc.read_usages.iter() {
					let r: ImageUsageFlags = read.usage.into();
					usage |= r;
					flags |= read.create_flags();
					usages.insert(
						pass,
						Usage {
							write: false,
							usage: read,
						},
					);

					if read.format != desc.write_usage.format {
						flags |= ImageCreateFlags::MUTABLE_FORMAT;
					}
				}

				ResourceDescType::Image(GpuResource {
					handle: ImageDesc {
						flags,
						format: desc.write_usage.format,
						size: desc.desc.size,
						levels: desc.desc.levels,
						layers: desc.desc.layers,
						samples: desc.desc.samples,
						usage,
					},
					usages,
				})
			},
		}
	}
}

impl<'graph> ResourceDescType<'graph> {
	unsafe fn gpu_buffer(&mut self) -> &mut GpuResource<'graph, BufferDesc, BufferUsage> {
		match self {
			ResourceDescType::GpuBuffer(res) => res,
			_ => unreachable_unchecked(),
		}
	}

	unsafe fn image(&mut self) -> &mut GpuResource<'graph, ImageDesc, ImageUsage> {
		match self {
			ResourceDescType::Image(res) => res,
			_ => unreachable_unchecked(),
		}
	}
}

struct ResourceDesc<'graph> {
	lifetime: ResourceLifetime,
	ty: ResourceDescType<'graph>,
}

impl<'graph> From<VirtualResourceData<'graph>> for ResourceDesc<'graph> {
	fn from(value: VirtualResourceData<'graph>) -> Self {
		Self {
			ty: value.ty.to_res(value.lifetime.start),
			lifetime: value.lifetime,
		}
	}
}

impl<'a> ResourceDesc<'a> {
	/// Returns `true` if the resource was merged.
	unsafe fn try_merge(&mut self, other: &VirtualResourceData<'a>) -> bool {
		if !self.lifetime.independent(other.lifetime) {
			return false;
		}

		let ret = match &other.ty {
			VirtualResourceType::Data(_) | VirtualResourceType::UploadBuffer(_) => unreachable_unchecked(),
			VirtualResourceType::GpuBuffer(desc) => {
				let this = self.ty.gpu_buffer();
				this.handle.size = this.handle.size.max(desc.desc);
				let u: BufferUsageFlags = desc.write_usage.into();
				this.handle.usage |= u;
				this.usages.insert(
					other.lifetime.start,
					Usage {
						write: true,
						usage: desc.write_usage,
					},
				);

				for (&pass, &read) in desc.read_usages.iter() {
					let r: BufferUsageFlags = read.into();
					this.handle.usage |= r;
					this.usages.insert(
						pass,
						Usage {
							write: false,
							usage: read,
						},
					);
				}

				true
			},
			VirtualResourceType::Image(desc) => {
				let this = self.ty.image();
				if !compatible_formats(this.handle.format, desc.write_usage.format) {
					return false;
				}

				let u: ImageUsageFlags = desc.write_usage.usage.into();
				this.handle.usage |= u;
				this.handle.flags |= desc.write_usage.create_flags();
				this.usages.insert(
					other.lifetime.start,
					Usage {
						write: true,
						usage: desc.write_usage,
					},
				);

				for (&pass, &read) in desc.read_usages.iter() {
					let r: ImageUsageFlags = read.usage.into();
					this.handle.usage |= r;
					this.handle.flags |= read.create_flags();
					this.usages.insert(
						pass,
						Usage {
							write: false,
							usage: read,
						},
					);
				}

				true
			},
		};

		if ret {
			self.lifetime = self.lifetime.union(other.lifetime);
		}
		ret
	}
}

struct ResourceAliaser<'graph> {
	aliasable: ArenaMap<'graph, MergeCandidate, Vec<u32, &'graph Arena>>,
	resource_map: Vec<u32, &'graph Arena>,
	resources: Vec<ResourceDesc<'graph>, &'graph Arena>,
}

impl<'graph> ResourceAliaser<'graph> {
	fn new(arena: &'graph Arena) -> Self {
		Self {
			aliasable: ArenaMap::with_hasher_in(BuildHasherDefault::default(), arena),
			resources: Vec::new_in(arena),
			resource_map: Vec::new_in(arena),
		}
	}

	fn push(&mut self, desc: ResourceDesc<'graph>) {
		self.resource_map.push(self.resources.len() as u32);
		self.resources.push(desc);
	}

	unsafe fn merge(&mut self, merge: MergeCandidate, resource: VirtualResourceData<'graph>) {
		let merges = self
			.aliasable
			.entry(merge)
			.or_insert(Vec::new_in(self.resources.allocator()));

		for &i in merges.iter() {
			let res = &mut self.resources[i as usize];
			if res.try_merge(&resource) {
				self.resource_map.push(i);
				return;
			}
		}

		merges.push(self.resources.len() as u32);
		self.push(resource.into());
	}

	fn add(&mut self, resource: VirtualResourceData<'graph>) {
		match &resource.ty {
			VirtualResourceType::Data(_) | VirtualResourceType::UploadBuffer(_) => self.push(resource.into()),
			VirtualResourceType::GpuBuffer(_) => unsafe { self.merge(MergeCandidate::GpuBuffer, resource) },
			VirtualResourceType::Image(desc) => unsafe { self.merge(MergeCandidate::Image(desc.desc), resource) },
		}
	}

	fn finish(self, device: &Device, graph: &mut RenderGraph) -> ResourceMap<'graph> {
		let alloc = *self.resources.allocator();
		let mut buffers = Vec::new_in(alloc);
		let mut images = Vec::new_in(alloc);

		let resources = self.resources.into_iter().enumerate().map(|(i, desc)| match desc.ty {
			ResourceDescType::Data(data) => Resource::Data(data, DataState::Uninit),
			ResourceDescType::UploadBuffer(desc) => Resource::UploadBuffer(
				graph.caches.upload_buffers[graph.curr_frame]
					.get(device, desc)
					.expect("failed to allocate upload buffer"),
			),
			ResourceDescType::GpuBuffer(desc) => {
				buffers.push(i as _);
				Resource::GpuBuffer(GpuResource {
					handle: graph
						.caches
						.gpu_buffers
						.get(device, desc.handle)
						.expect("failed to allocate gpu buffer"),
					usages: desc.usages,
				})
			},
			ResourceDescType::Image(desc) => {
				images.push(i as _);
				Resource::Image(GpuResource {
					handle: graph
						.caches
						.images
						.get(device, desc.handle)
						.expect("failed to allocate image"),
					usages: desc.usages,
				})
			},
		});

		unsafe { ResourceMap::new(self.resource_map, resources.collect_in(alloc), buffers, images) }
	}
}

trait AccessExt: Sized {
	fn merge(&self, other: &Self) -> Option<Self>;
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Default)]
pub struct Access {
	pub stage: PipelineStageFlags2,
	pub access: AccessFlags2,
}

impl AccessExt for Access {
	fn merge(&self, other: &Self) -> Option<Self> {
		Some(Self {
			stage: self.stage | other.stage,
			access: self.access | other.access,
		})
	}
}

#[derive(Copy, Clone, Default)]
struct ImageAccess {
	pub access: Access,
	pub layout: ImageLayout,
	pub format: Format,
}

impl AccessExt for ImageAccess {
	fn merge(&self, other: &Self) -> Option<Self> {
		if self.layout != other.layout {
			return None;
		}

		Some(Self {
			access: self.access.merge(&other.access)?,
			layout: self.layout,
			format: self.format,
		})
	}
}

fn image_stage_mask(usage: ImageUsage) -> PipelineStageFlags2 {
	match usage.usage {
		ImageUsageType::TransferSrc | ImageUsageType::TransferDst => PipelineStageFlags2::TRANSFER,
		ImageUsageType::Sampled(s) | ImageUsageType::Storage(s) => s.into(),
		ImageUsageType::ColorAttachment | ImageUsageType::DepthStencilAttachment => {
			PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
		},
	}
}

fn image_access_mask(usage: ImageUsage, write: bool) -> AccessFlags2 {
	match usage.usage {
		ImageUsageType::TransferSrc => AccessFlags2::TRANSFER_READ,
		ImageUsageType::TransferDst => AccessFlags2::TRANSFER_WRITE,
		ImageUsageType::Sampled(_) => AccessFlags2::SHADER_SAMPLED_READ,
		ImageUsageType::Storage(_) => {
			if write {
				AccessFlags2::SHADER_STORAGE_READ | AccessFlags2::SHADER_STORAGE_WRITE
			} else {
				AccessFlags2::SHADER_STORAGE_READ
			}
		},
		ImageUsageType::ColorAttachment => AccessFlags2::COLOR_ATTACHMENT_WRITE,
		ImageUsageType::DepthStencilAttachment => AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
	}
}

fn image_layout(usage: ImageUsage) -> ImageLayout {
	match usage.usage {
		ImageUsageType::TransferSrc => ImageLayout::TRANSFER_SRC_OPTIMAL,
		ImageUsageType::TransferDst => ImageLayout::TRANSFER_DST_OPTIMAL,
		ImageUsageType::Sampled(_) => sampled_image_layout(usage.format),
		ImageUsageType::Storage(_) => ImageLayout::GENERAL,
		ImageUsageType::ColorAttachment => ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
		ImageUsageType::DepthStencilAttachment => match usage.format {
			Format::D16_UNORM | Format::X8_D24_UNORM_PACK32 | Format::D32_SFLOAT => {
				ImageLayout::DEPTH_ATTACHMENT_OPTIMAL
			},
			Format::D16_UNORM_S8_UINT | Format::D24_UNORM_S8_UINT | Format::D32_SFLOAT_S8_UINT => {
				ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
			},
			Format::S8_UINT => ImageLayout::STENCIL_ATTACHMENT_OPTIMAL,
			_ => unreachable!("unsupported depth format: {:?}", usage.format),
		},
	}
}

fn buffer_stage_mask(usage: BufferUsage) -> PipelineStageFlags2 {
	match usage {
		BufferUsage::TransferSrc | BufferUsage::TransferDst => PipelineStageFlags2::TRANSFER,
		BufferUsage::Storage(s) => s.into(),
		BufferUsage::Index => PipelineStageFlags2::INDEX_INPUT,
		BufferUsage::Vertex => PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT,
		BufferUsage::Indirect => PipelineStageFlags2::DRAW_INDIRECT,
	}
}

fn buffer_access_mask(usage: BufferUsage, write: bool) -> AccessFlags2 {
	match usage {
		BufferUsage::TransferSrc => AccessFlags2::TRANSFER_READ,
		BufferUsage::TransferDst => AccessFlags2::TRANSFER_WRITE,
		BufferUsage::Storage(_) => {
			if write {
				AccessFlags2::SHADER_STORAGE_READ | AccessFlags2::SHADER_STORAGE_WRITE
			} else {
				AccessFlags2::SHADER_STORAGE_READ
			}
		},
		BufferUsage::Index => AccessFlags2::INDEX_READ,
		BufferUsage::Vertex => AccessFlags2::VERTEX_ATTRIBUTE_READ,
		BufferUsage::Indirect => AccessFlags2::INDIRECT_COMMAND_READ,
	}
}

fn image_access(usage: ImageUsage, write: bool) -> ImageAccess {
	ImageAccess {
		access: Access {
			stage: image_stage_mask(usage),
			access: image_access_mask(usage, write),
		},
		layout: image_layout(usage),
		format: usage.format,
	}
}

fn buffer_access(usage: BufferUsage, write: bool) -> Access {
	Access {
		stage: buffer_stage_mask(usage),
		access: buffer_access_mask(usage, write),
	}
}

fn image_barrier(
	image: Image, prev: ImageAccess, curr: ImageAccess, format: Format, write: bool,
) -> ImageMemoryBarrier2 {
	ImageMemoryBarrier2::builder()
		.image(image)
		.subresource_range(
			ImageSubresourceRange::builder()
				.base_mip_level(0)
				.base_array_layer(0)
				.level_count(REMAINING_MIP_LEVELS)
				.layer_count(REMAINING_ARRAY_LAYERS)
				.aspect_mask(image_aspect_mask(format))
				.build(),
		)
		.src_stage_mask(prev.access.stage)
		.src_access_mask(prev.access.access)
		.old_layout(if write { ImageLayout::UNDEFINED } else { prev.layout })
		.dst_stage_mask(curr.access.stage)
		.dst_access_mask(curr.access.access)
		.new_layout(curr.layout)
		.build()
}

struct MergeReads<T: Iterator> {
	iter: Peekable<T>,
}

impl<T: Iterator<Item = (u32, Usage<U>)>, U> MergeReads<T> {
	fn new(iter: T) -> Self { Self { iter: iter.peekable() } }
}

impl<'a, T: Iterator<Item = (u32, Usage<U>)>, U: AccessExt> Iterator for MergeReads<T> {
	type Item = (u32, Usage<U>);

	fn next(&mut self) -> Option<Self::Item> {
		match self.iter.next()? {
			x @ (_, Usage { write: true, .. }) => Some(x),
			(
				pass,
				Usage {
					write: false,
					mut usage,
				},
			) => {
				while let Some((_, next_usage)) = self.iter.peek() {
					if !next_usage.write {
						if let Some(u) = usage.merge(&next_usage.usage) {
							usage = u;
							self.iter.next();
						} else {
							break;
						}
					} else {
						break;
					}
				}
				Some((pass, Usage { write: false, usage }))
			},
		}
	}
}

pub struct Synchronizer<'temp, 'graph> {
	resource_map: &'temp ResourceMap<'graph>,
	passes: &'temp [PassData<'temp, 'graph>],
}

impl<'temp, 'graph> Synchronizer<'temp, 'graph> {
	fn new(resource_map: &'temp ResourceMap<'graph>, passes: &'temp [PassData<'temp, 'graph>]) -> Self {
		Self { resource_map, passes }
	}

	fn sync(&mut self) -> Vec<Synchronization<'graph>, &'graph Arena> {
		let mut sync: Vec<_, _> = std::iter::repeat_with(|| Synchronization {
			barriers: ArenaMap::with_hasher_in(Default::default(), self.resource_map.arena()),
			image_barriers: Vec::new_in(self.resource_map.arena()),
		})
		.take(self.passes.len())
		.collect_in(self.resource_map.arena());

		for buffer in self.resource_map.buffers() {
			let mut prev_access = Access::default();
			for (pass, access) in MergeReads::new(buffer.usages.iter().map(|(&x, &y)| {
				(
					x,
					Usage {
						write: y.write,
						usage: buffer_access(y.usage, y.write),
					},
				)
			})) {
				let barrier = sync[pass as usize].barriers.entry(prev_access).or_default();
				*barrier = barrier.merge(&access.usage).unwrap();
				prev_access = access.usage;
			}
		}

		for image in self.resource_map.images() {
			let mut prev_access = ImageAccess::default();
			for (pass, access) in MergeReads::new(image.usages.iter().map(|(&x, &y)| {
				(
					x,
					Usage {
						write: y.write,
						usage: image_access(y.usage, y.write),
					},
				)
			})) {
				let sync = &mut sync[pass as usize];
				if prev_access.layout == access.usage.layout {
					if let Some(barrier) = sync.barriers.get_mut(&prev_access.access) {
						// If there's no layout transition and an existing memory barrier, merge ourselves with it.
						*barrier = barrier.merge(&access.usage.access).unwrap();
						prev_access.access = *barrier;
						continue;
					}
				}

				sync.image_barriers.push(image_barrier(
					image.handle,
					prev_access,
					access.usage,
					access.usage.format,
					access.write,
				));
				prev_access = access.usage;
			}
		}

		sync
	}
}

impl<'pass, 'graph> Frame<'pass, 'graph> {
	pub(super) fn compile(self, device: &Device) -> Result<CompiledFrame<'pass, 'graph>> {
		let span = span!(Level::TRACE, "compile graph");
		let _e = span.enter();

		// The order of the passes is already topologically sorted.
		// This is order we will run them in.

		let resource_map = {
			let span = span!(Level::TRACE, "alias resources");
			let _e = span.enter();

			self.virtual_resources
				.into_iter()
				.fold(ResourceAliaser::new(self.arena), |mut a, r| {
					a.add(r);
					a
				})
				.finish(device, self.graph)
		};

		let sync = {
			let span = span!(Level::TRACE, "synchronize");
			let _e = span.enter();

			Synchronizer::new(&resource_map, &self.passes).sync()
		};

		Ok(CompiledFrame {
			passes: self.passes,
			sync,
			resource_map,
			graph: self.graph,
		})
	}
}
