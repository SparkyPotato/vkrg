use std::{
	alloc::{Allocator, Layout},
	hint::unreachable_unchecked,
	marker::PhantomData,
	ops::RangeInclusive,
	ptr::NonNull,
};

use ash::vk::{
	BufferUsageFlags,
	CommandBuffer,
	CommandBufferSubmitInfo,
	DependencyInfo,
	Extent3D,
	Fence,
	Format,
	ImageUsageFlags,
	ImageViewType,
	SampleCountFlags,
	SemaphoreSubmitInfo,
	SubmitInfo2,
};
use tracing::{span, Level};

pub use crate::graph::resource::{GpuBufferHandle, ImageView, UploadBufferDesc, UploadBufferHandle};
use crate::{
	arena::{collect_allocated_vec, Arena},
	device::Device,
	graph::{
		cache::{ResourceCache, UniqueCache},
		compile::Resource,
		frame_data::FrameData,
		resource::{GpuBuffer, Image, ImageFlags, ImageViewDesc, ImageViewUsage, UploadBuffer},
	},
	Result,
};

pub mod cache;
mod compile;
mod frame_data;
pub mod resource;
#[cfg(test)]
mod test;

/// The render graph.
pub struct RenderGraph {
	frame_data: [FrameData; 2],
	caches: Caches,
	curr_frame: usize,
	resource_base_id: usize,
}

#[doc(hidden)]
pub struct Caches {
	upload_buffers: [ResourceCache<UploadBuffer>; 2],
	gpu_buffers: ResourceCache<GpuBuffer>,
	images: ResourceCache<Image>,
	image_views: UniqueCache<ImageView>,
}

impl RenderGraph {
	pub fn new(device: &Device) -> Result<Self> {
		let frame_data = [FrameData::new(device)?, FrameData::new(device)?];

		let caches = Caches {
			upload_buffers: [ResourceCache::new(), ResourceCache::new()],
			gpu_buffers: ResourceCache::new(),
			images: ResourceCache::new(),
			image_views: UniqueCache::new(),
		};

		Ok(Self {
			frame_data,
			caches,
			curr_frame: 0,
			resource_base_id: 0,
		})
	}

	pub fn frame<'pass, 'graph>(&'graph mut self, arena: &'graph Arena) -> Frame<'pass, 'graph> {
		Frame {
			graph: self,
			arena,
			passes: Vec::new_in(arena),
			virtual_resources: Vec::new_in(arena),
		}
	}

	pub fn destroy(self, device: &Device) {
		for frame_data in self.frame_data {
			frame_data.destroy(device);
		}
		for cache in self.caches.upload_buffers {
			cache.destroy(device);
		}
		self.caches.gpu_buffers.destroy(device);
		self.caches.images.destroy(device);
		self.caches.image_views.destroy(device);
	}

	fn next_frame(&mut self, resource_count: usize) {
		self.curr_frame ^= 1;
		self.resource_base_id = self.resource_base_id.wrapping_add(resource_count);
	}
}

pub struct Frame<'pass, 'graph> {
	graph: &'graph mut RenderGraph,
	arena: &'graph Arena,
	passes: Vec<PassData<'pass, 'graph>, &'graph Arena>,
	virtual_resources: Vec<VirtualResourceData<'graph>, &'graph Arena>,
}

impl<'pass, 'graph> Frame<'pass, 'graph> {
	pub fn pass<'frame>(&'frame mut self, name: &str) -> PassBuilder<'frame, 'pass, 'graph> {
		let arena = self.arena;
		let name = name.as_bytes().iter().copied().chain([0]);
		PassBuilder {
			name: collect_allocated_vec(name, arena),
			frame: self,
			inputs: Vec::new_in(arena),
			inner: Vec::new_in(arena),
			outputs: Vec::new_in(arena),
			topology: 0,
		}
	}

	pub fn run(self, device: &Device) -> Result<()> {
		let arena = self.arena;
		let mut compiled = self.compile(device);
		let graph = compiled.graph;
		let data = &mut graph.frame_data[graph.curr_frame];

		data.reset(device)?;
		unsafe {
			graph.caches.upload_buffers[graph.curr_frame].reset(device);
			graph.caches.gpu_buffers.reset(device);
			graph.caches.images.reset(device);
			graph.caches.image_views.reset(device);
		}

		let buf = data.cmd_buf(device)?;

		for (i, pass) in compiled.passes.into_iter().enumerate() {
			{
				let span = span!(
					Level::TRACE,
					"run pass",
					name = unsafe { &std::str::from_utf8_unchecked(&pass.name[..pass.name.len() - 1]) }
				);
				let _e = span.enter();

				let sync = pass.sync;
				unsafe {
					device.device().cmd_pipeline_barrier2(
						buf,
						&DependencyInfo::builder()
							.memory_barriers(&sync.barriers)
							.image_memory_barriers(&sync.image_barriers),
					);
				}

				(pass.exec)(PassContext {
					arena,
					device,
					buf,
					base_id: graph.resource_base_id,
					pass: i,
					resource_map: &compiled.resource_map,
					resources: &mut compiled.resources,
					caches: &mut graph.caches,
				});
			}
		}

		let (sem, set) = data.semaphore.next();
		unsafe {
			let other_frame = &graph.frame_data[graph.curr_frame ^ 1];
			device.submit_graphics(
				&[SubmitInfo2::builder()
					.wait_semaphore_infos(&[SemaphoreSubmitInfo::builder()
						.semaphore(other_frame.semaphore.semaphore())
						.value(other_frame.semaphore.value())
						.build()])
					.command_buffer_infos(&[CommandBufferSubmitInfo::builder().command_buffer(buf).build()])
					.signal_semaphore_infos(&[SemaphoreSubmitInfo::builder().semaphore(sem).value(set).build()])
					.build()],
				Fence::null(),
			)?;
		}

		graph.next_frame(compiled.resources.len());

		Ok(())
	}
}

pub struct PassBuilder<'frame, 'pass, 'graph> {
	name: Vec<u8, &'graph Arena>,
	frame: &'frame mut Frame<'pass, 'graph>,
	inputs: Vec<usize, &'graph Arena>,
	inner: Vec<usize, &'graph Arena>,
	outputs: Vec<usize, &'graph Arena>,
	topology: usize,
}

impl<'frame, 'pass, 'graph> PassBuilder<'frame, 'pass, 'graph> {
	/// Create an input from another pass' output.
	pub fn input<T: VirtualResource>(&mut self, id: ReadId<T>, usage: <T::Desc as VirtualResourceDesc>::Usage) {
		let id = id.id.wrapping_sub(self.frame.graph.resource_base_id);

		self.inputs.push(id);
		unsafe {
			let res = self.frame.virtual_resources.get_unchecked_mut(id);
			let source_topo = self.frame.passes.get_unchecked(res.pass_written).topology;
			self.topology = self.topology.max(source_topo + 1);
			T::Desc::add_read_usage(&mut res.ty, usage, self.frame.passes.len());
		}
	}

	/// Create an pass-local transient resource.
	pub fn inner<D: InnerResourceDesc>(&mut self, desc: D) -> InnerId<D::Resource> {
		let real_id = self.frame.virtual_resources.len();
		let id = real_id.wrapping_add(self.frame.graph.resource_base_id);

		let ty = desc.ty(self.frame.arena);

		self.inner.push(real_id);
		self.frame.virtual_resources.push(VirtualResourceData {
			ty,
			pass_written: self.frame.passes.len(),
		});

		InnerId {
			id,
			_marker: PhantomData,
		}
	}

	/// This pass outputs some GPU data for other passes.
	pub fn output<D: VirtualResourceDesc>(
		&mut self, desc: D, usage: D::Usage,
	) -> (ReadId<D::Resource>, WriteId<D::Resource>) {
		let real_id = self.frame.virtual_resources.len();
		let id = real_id.wrapping_add(self.frame.graph.resource_base_id);

		let ty = desc.ty(usage, self.frame.arena);
		debug_assert!(
			!matches!(ty, VirtualResourceType::UploadBuffer(_)),
			"Upload buffers can't be outputs"
		);

		self.outputs.push(real_id);
		self.frame.virtual_resources.push(VirtualResourceData {
			ty,
			pass_written: self.frame.passes.len(),
		});

		(
			ReadId {
				id,
				_marker: PhantomData,
			},
			WriteId {
				id,
				_marker: PhantomData,
			},
		)
	}

	/// Create an input from another pass' CPU data.
	pub fn data_input<T>(&mut self, id: &GetId<T>) { self.data_input_inner(id.id); }

	/// Just like [`Self::data_input`], but the pass only gets a reference to the data.
	pub fn data_input_ref<T>(&mut self, id: RefId<T>) { self.data_input_inner(id.id); }

	fn data_input_inner(&mut self, id: usize) {
		let id = id.wrapping_sub(self.frame.graph.resource_base_id);

		self.inputs.push(id);
		unsafe {
			let res = self.frame.virtual_resources.get_unchecked(id);
			let source_topo = self.frame.passes.get_unchecked(res.pass_written).topology;
			self.topology = self.topology.max(source_topo + 1);
		}
	}

	/// This pass outputs some CPU data for other passes.
	pub fn data_output<T>(&mut self) -> (SetId<T>, GetId<T>) {
		let real_id = self.frame.virtual_resources.len();
		let id = real_id.wrapping_add(self.frame.graph.resource_base_id);

		self.outputs.push(real_id);
		self.frame.virtual_resources.push(VirtualResourceData {
			pass_written: self.frame.passes.len(),
			ty: VirtualResourceType::Data(self.frame.arena.allocate(Layout::new::<T>()).unwrap().cast()),
		});

		(
			SetId {
				id,
				_marker: PhantomData,
			},
			GetId {
				id,
				_marker: PhantomData,
			},
		)
	}

	pub fn build(self, callback: impl FnOnce(PassContext) + 'pass) {
		let pass = PassData {
			name: self.name,
			callback: Box::new_in(callback, self.frame.arena),
			inputs: self.inputs,
			inner: self.inner,
			outputs: self.outputs,
			topology: self.topology,
		};
		self.frame.passes.push(pass);
	}
}

pub struct PassContext<'frame, 'graph> {
	pub arena: &'graph Arena,
	pub device: &'frame Device,
	pub buf: CommandBuffer,
	base_id: usize,
	pass: usize,
	resource_map: &'frame Vec<usize, &'graph Arena>,
	resources: &'frame mut Vec<Resource<'graph>, &'graph Arena>,
	caches: &'frame mut Caches,
}

impl<'frame, 'graph> PassContext<'frame, 'graph> {
	pub fn get_data_ref<T: 'frame>(&self, id: RefId<T>) -> &'frame T {
		let id = id.id.wrapping_sub(self.base_id);
		assert!(id < self.resources.len(), "RefId from previous frame used");
		unsafe {
			let res = self.resources.get_unchecked(*self.resource_map.get_unchecked(id));
			let (ptr, init) = res.data();

			assert!(init, "Transient Data has not been initialized");
			&*ptr.as_ptr()
		}
	}

	pub fn get_data<T: 'frame>(&self, id: GetId<T>) -> T {
		let id = id.id.wrapping_sub(self.base_id);
		assert!(id < self.resources.len(), "GetId from previous frame used");
		unsafe {
			let res = self.resources.get_unchecked(*self.resource_map.get_unchecked(id));
			let (ptr, init) = res.data::<T>();

			assert!(init, "Transient Data has not been initialized");
			ptr.as_ptr().read()
		}
	}

	pub fn set_data<T: 'frame>(&mut self, id: SetId<T>, data: T) {
		let id = id.id.wrapping_sub(self.base_id);
		assert!(id < self.resources.len(), "SetId from previous frame used");
		unsafe {
			let res = self.resources.get_unchecked_mut(*self.resource_map.get_unchecked(id));

			res.data::<T>().0.as_ptr().write(data);
			res.init();
		}
	}

	pub fn read<T: VirtualResource>(&mut self, id: ReadId<T>) -> T {
		let id = id.id.wrapping_sub(self.base_id);
		assert!(id < self.resources.len(), "ReadId from previous frame used");
		unsafe {
			let res = self.resources.get_unchecked(*self.resource_map.get_unchecked(id));
			T::from_res(self.pass, res, &mut self.caches, &self.device)
		}
	}

	pub fn write<T: VirtualResource>(&mut self, id: WriteId<T>) -> T {
		let id = id.id.wrapping_sub(self.base_id);
		assert!(id < self.resources.len(), "WriteId from previous frame used");
		unsafe {
			let res = self.resources.get_unchecked_mut(*self.resource_map.get_unchecked(id));
			T::from_res(self.pass, res, &mut self.caches, &self.device)
		}
	}

	pub fn inner<T: InnerResource>(&mut self, id: InnerId<T>) -> T {
		let id = id.id.wrapping_sub(self.base_id);
		assert!(id < self.resources.len(), "InnerId from previous frame used");
		unsafe {
			let res = self.resources.get_unchecked_mut(*self.resource_map.get_unchecked(id));
			T::from_res(res)
		}
	}
}

pub struct SetId<T> {
	id: usize,
	_marker: PhantomData<T>,
}

#[derive(Copy, Clone)]
pub struct GetId<T> {
	id: usize,
	_marker: PhantomData<T>,
}

impl<T> GetId<T> {
	pub fn to_ref(self) -> RefId<T> {
		RefId {
			id: self.id,
			_marker: PhantomData,
		}
	}
}

pub struct RefId<T> {
	id: usize,
	_marker: PhantomData<T>,
}

impl<T> Copy for RefId<T> {}

impl<T> Clone for RefId<T> {
	fn clone(&self) -> Self { *self }
}

impl<'frame, T> From<GetId<T>> for RefId<T> {
	fn from(id: GetId<T>) -> Self { id.to_ref() }
}

pub struct WriteId<T: VirtualResource> {
	id: usize,
	_marker: PhantomData<T>,
}

pub struct ReadId<T: VirtualResource> {
	id: usize,
	_marker: PhantomData<T>,
}

impl<T: VirtualResource> Copy for ReadId<T> {}

impl<T: VirtualResource> Clone for ReadId<T> {
	fn clone(&self) -> Self { *self }
}

pub struct InnerId<T: InnerResource> {
	id: usize,
	_marker: PhantomData<T>,
}

#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct GpuBufferDesc {
	pub size: usize,
}

#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct GpuBufferUsage {
	pub usage: BufferUsageFlags,
}

#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct ImageDesc {
	pub size: Extent3D,
	pub levels: u32,
	pub layers: u32,
	pub samples: SampleCountFlags,
}

#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct ImageUsage {
	pub format: Format,
	pub usage: ImageUsageFlags,
	pub shader_usage: ImageViewUsage,
	pub view_type: ImageViewType,
}

struct PassData<'pass, 'graph> {
	// UTF-8 encoded, null terminated.
	name: Vec<u8, &'graph Arena>,
	callback: Box<dyn FnOnce(PassContext) + 'pass, &'graph Arena>,
	// Indices of the virtual resources used by this pass.
	inputs: Vec<usize, &'graph Arena>,
	inner: Vec<usize, &'graph Arena>,
	outputs: Vec<usize, &'graph Arena>,
	// The index of the topology of this pass.
	// All passes with the same topology run at the same time.
	topology: usize,
}

#[derive(Clone)]
struct VirtualResourceData<'graph> {
	pass_written: usize,
	ty: VirtualResourceType<'graph>,
}

impl VirtualResourceData<'_> {
	fn lifetime(&self) -> RangeInclusive<usize> {
		match &self.ty {
			VirtualResourceType::Data(_) => self.pass_written..=self.pass_written,
			VirtualResourceType::UploadBuffer(_) => self.pass_written..=self.pass_written,
			VirtualResourceType::GpuBuffer { read_usages, .. } => {
				self.pass_written..=read_usages.last().map(|x| x.pass).unwrap_or(self.pass_written)
			},
			VirtualResourceType::Image { read_usages, .. } => {
				self.pass_written..=read_usages.last().map(|x| x.pass).unwrap_or(self.pass_written)
			},
		}
	}
}

/// This trait is sealed.
pub trait VirtualResourceDesc: sealed::Sealed {
	type Resource: VirtualResource;
	type Usage;

	fn ty(self, usage: Self::Usage, arena: &Arena) -> VirtualResourceType;

	unsafe fn add_read_usage(ty: &mut VirtualResourceType, usage: Self::Usage, pass: usize);
}

/// This trait is sealed.
pub trait VirtualResource: sealed::Sealed {
	type Desc: VirtualResourceDesc;

	unsafe fn from_res(pass: usize, res: &Resource, caches: &mut Caches, device: &Device) -> Self;
}

/// This trait is sealed.
pub trait InnerResourceDesc: sealed::Sealed {
	type Resource: InnerResource;

	fn ty(self, arena: &Arena) -> VirtualResourceType;
}

/// This trait is sealed.
pub trait InnerResource: sealed::Sealed {
	type Desc: InnerResourceDesc;

	unsafe fn from_res(res: &Resource) -> Self;
}

#[doc(hidden)]
#[derive(Clone, Hash, PartialEq, Eq)]
pub enum VirtualResourceType<'graph> {
	Data(NonNull<()>),
	UploadBuffer(UploadBufferDesc),
	GpuBuffer {
		desc: resource::GpuBufferDesc,
		read_usages: Vec<ReadUsage<()>, &'graph Arena>,
	},
	Image {
		desc: resource::ImageDesc,
		write_usage: InnerUsage,
		read_usages: Vec<ReadUsage<InnerUsage>, &'graph Arena>,
	},
}

#[doc(hidden)]
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct InnerUsage {
	view_type: ImageViewType,
	format: Format,
	shader_usage: ImageViewUsage,
}

#[doc(hidden)]
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct ReadUsage<T> {
	pass: usize,
	usage: T,
}

impl<'graph> VirtualResourceType<'graph> {
	unsafe fn gpu_buffer_usage(&mut self) -> &mut BufferUsageFlags {
		match self {
			VirtualResourceType::GpuBuffer { desc, .. } => &mut desc.usage,
			_ => unreachable_unchecked(),
		}
	}

	unsafe fn gpu_buffer_read_usages(&mut self) -> &mut Vec<ReadUsage<()>, &'graph Arena> {
		match self {
			VirtualResourceType::GpuBuffer { read_usages, .. } => read_usages,
			_ => unreachable_unchecked(),
		}
	}

	unsafe fn image_desc(&mut self) -> &mut resource::ImageDesc {
		match self {
			VirtualResourceType::Image { desc, .. } => desc,
			_ => unreachable_unchecked(),
		}
	}

	unsafe fn image_write_usage(&mut self) -> &mut InnerUsage {
		match self {
			VirtualResourceType::Image { write_usage, .. } => write_usage,
			_ => unreachable_unchecked(),
		}
	}

	unsafe fn image_read_usages(&mut self) -> &mut Vec<ReadUsage<InnerUsage>, &'graph Arena> {
		match self {
			VirtualResourceType::Image { read_usages, .. } => read_usages,
			_ => unreachable_unchecked(),
		}
	}
}

impl InnerResourceDesc for UploadBufferDesc {
	type Resource = UploadBufferHandle;

	fn ty(self, _: &Arena) -> VirtualResourceType { VirtualResourceType::UploadBuffer(self) }
}

impl InnerResource for UploadBufferHandle {
	type Desc = UploadBufferDesc;

	unsafe fn from_res(res: &Resource) -> Self { res.upload_buffer() }
}

impl VirtualResourceDesc for GpuBufferDesc {
	type Resource = GpuBufferHandle;
	type Usage = GpuBufferUsage;

	fn ty(self, usage: GpuBufferUsage, arena: &Arena) -> VirtualResourceType {
		VirtualResourceType::GpuBuffer {
			desc: resource::GpuBufferDesc {
				size: self.size,
				usage: usage.usage,
			},
			read_usages: Vec::new_in(arena),
		}
	}

	unsafe fn add_read_usage(ty: &mut VirtualResourceType, usage: Self::Usage, pass: usize) {
		*ty.gpu_buffer_usage() |= usage.usage;
		ty.gpu_buffer_read_usages().push(ReadUsage { usage: (), pass })
	}
}

impl VirtualResource for GpuBufferHandle {
	type Desc = GpuBufferDesc;

	unsafe fn from_res(_: usize, res: &Resource, _: &mut Caches, _: &Device) -> Self { res.gpu_buffer() }
}

impl VirtualResourceDesc for ImageDesc {
	type Resource = ImageView;
	type Usage = ImageUsage;

	fn ty(self, usage: ImageUsage, arena: &Arena) -> VirtualResourceType {
		VirtualResourceType::Image {
			desc: resource::ImageDesc {
				flags: ImageFlags::None,
				format: usage.format,
				size: self.size,
				levels: self.levels,
				layers: self.layers,
				samples: self.samples,
				usage: usage.usage,
			},
			write_usage: InnerUsage {
				view_type: usage.view_type,
				format: usage.format,
				shader_usage: usage.shader_usage,
			},
			read_usages: Vec::new_in(arena),
		}
	}

	unsafe fn add_read_usage(ty: &mut VirtualResourceType, usage: Self::Usage, pass: usize) {
		let desc = ty.image_desc();

		desc.usage |= usage.usage;
		let flags = match usage.view_type {
			ImageViewType::TYPE_1D | ImageViewType::TYPE_2D | ImageViewType::TYPE_3D => ImageFlags::None,
			ImageViewType::CUBE => ImageFlags::Cube,
			ImageViewType::TYPE_1D_ARRAY | ImageViewType::TYPE_2D_ARRAY => ImageFlags::Array,
			ImageViewType::CUBE_ARRAY => ImageFlags::CubeAndArray,
			_ => unreachable!(),
		};
		desc.flags = match (desc.flags, flags) {
			(ImageFlags::None, x) => x,
			(x, ImageFlags::None) => x,
			(x, y) if x == y => x,
			(ImageFlags::Array, ImageFlags::Cube) => ImageFlags::CubeAndArray,
			(ImageFlags::Cube, ImageFlags::Array) => ImageFlags::CubeAndArray,
			(_, ImageFlags::CubeAndArray) => ImageFlags::CubeAndArray,
			(ImageFlags::CubeAndArray, _) => ImageFlags::CubeAndArray,
			_ => unreachable!(),
		};
		ty.image_read_usages().push(ReadUsage {
			pass,
			usage: InnerUsage {
				view_type: usage.view_type,
				format: usage.format,
				shader_usage: usage.shader_usage,
			},
		});
	}
}

impl VirtualResource for ImageView {
	type Desc = ImageDesc;

	unsafe fn from_res(pass: usize, res: &Resource, caches: &mut Caches, device: &Device) -> Self {
		let (image, usages) = res.image();
		let usage = usages.iter().find(|x| x.pass == pass).unwrap().usage;
		caches
			.image_views
			.get(
				device,
				ImageViewDesc {
					image,
					view_type: usage.view_type,
					format: usage.format,
					usage: usage.shader_usage,
				},
			)
			.expect("Failed to create image view")
	}
}

mod sealed {
	use crate::graph::{GpuBufferDesc, GpuBufferHandle, ImageDesc, ImageView, UploadBufferDesc, UploadBufferHandle};

	pub trait Sealed {}

	impl Sealed for UploadBufferHandle {}
	impl Sealed for UploadBufferDesc {}

	impl Sealed for GpuBufferHandle {}
	impl Sealed for GpuBufferDesc {}

	impl Sealed for ImageDesc {}
	impl Sealed for ImageView {}
}
