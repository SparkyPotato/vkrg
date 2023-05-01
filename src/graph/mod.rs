use std::{
	alloc::{Allocator, Layout},
	hash::BuildHasherDefault,
	marker::PhantomData,
};

use ash::vk::{
	CommandBuffer,
	CommandBufferSubmitInfo,
	DependencyInfo,
	Fence,
	MemoryBarrier2,
	SemaphoreSubmitInfo,
	SubmitInfo2,
};
use hashbrown::HashMap;
use rustc_hash::FxHasher;
use tracing::{span, Level};

pub use crate::graph::virtual_resource::{GpuBufferDesc, ImageDesc, ImageUsage};
use crate::{
	arena::{Arena, IteratorAlloc},
	device::Device,
	graph::{
		cache::{ResourceCache, UniqueCache},
		compile::{DataState, ResourceMap},
		frame_data::FrameData,
		virtual_resource::{
			ResourceLifetime,
			VirtualResource,
			VirtualResourceData,
			VirtualResourceDesc,
			VirtualResourceType,
		},
	},
	resource::{GpuBuffer, GpuBufferHandle, Image, ImageView, UploadBuffer, UploadBufferHandle},
	Result,
};

pub mod cache;
mod compile;
mod frame_data;
mod virtual_resource;

const FRAMES_IN_FLIGHT: usize = 2;

/// The render graph.
pub struct RenderGraph {
	frame_data: [FrameData; FRAMES_IN_FLIGHT],
	caches: Caches,
	curr_frame: usize,
	resource_base_id: usize,
}

pub struct Caches {
	upload_buffers: [ResourceCache<UploadBuffer>; FRAMES_IN_FLIGHT],
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
		self.caches.image_views.destroy(device);
		self.caches.images.destroy(device);
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
	pub fn pass(&mut self, name: &str) -> PassBuilder<'_, 'pass, 'graph> {
		let arena = self.arena;
		let name = name.as_bytes().iter().copied().chain([0]);
		PassBuilder {
			name: name.collect_in(arena),
			frame: self,
		}
	}

	pub fn run(self, device: &Device) -> Result<()> {
		let arena = self.arena;
		let data = &mut self.graph.frame_data[self.graph.curr_frame];
		data.reset(device)?;
		unsafe {
			self.graph.caches.upload_buffers[self.graph.curr_frame].reset(device);
			self.graph.caches.gpu_buffers.reset(device);
			self.graph.caches.image_views.reset(device);
			self.graph.caches.images.reset(device);
		}

		let mut compiled = self.compile(device)?;

		let graph = compiled.graph;
		let data = &mut graph.frame_data[graph.curr_frame];
		let buf = data.cmd_buf(device)?;

		for (i, (pass, sync)) in compiled.passes.into_iter().zip(compiled.sync).enumerate() {
			{
				let name = unsafe { std::str::from_utf8_unchecked(&pass.name[..pass.name.len() - 1]) };
				let span = span!(Level::TRACE, "run pass", name = name);
				let _e = span.enter();

				let barriers: Vec<_, _> = sync
					.barriers
					.into_iter()
					.map(|(from, to)| {
						MemoryBarrier2::builder()
							.src_stage_mask(from.stage)
							.src_access_mask(from.access)
							.dst_stage_mask(to.stage)
							.dst_access_mask(to.access)
							.build()
					})
					.collect_in(arena);

				unsafe {
					device.device().cmd_pipeline_barrier2(
						buf,
						&DependencyInfo::builder()
							.memory_barriers(&barriers)
							.image_memory_barriers(&sync.image_barriers),
					);
				}

				(pass.callback)(PassContext {
					arena,
					device,
					buf,
					base_id: graph.resource_base_id,
					pass: i as u32,
					resource_map: &mut compiled.resource_map,
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

		let len = compiled.resource_map.cleanup();
		graph.next_frame(len);

		Ok(())
	}
}

pub struct PassBuilder<'frame, 'pass, 'graph> {
	name: Vec<u8, &'graph Arena>,
	frame: &'frame mut Frame<'pass, 'graph>,
}

impl<'frame, 'pass, 'graph> PassBuilder<'frame, 'pass, 'graph> {
	/// Read GPU data that another pass outputs.
	pub fn input<T: VirtualResource>(&mut self, id: ReadId<T>, usage: <T::Desc as VirtualResourceDesc>::Usage) {
		let id = id.id.wrapping_sub(self.frame.graph.resource_base_id);

		unsafe {
			let res = self.frame.virtual_resources.get_unchecked_mut(id);
			res.lifetime.end = self.frame.passes.len() as _;
			T::Desc::add_read_usage(res, self.frame.passes.len() as _, usage);
		}
	}

	/// Output GPU data for other passes.
	pub fn output<D: VirtualResourceDesc>(
		&mut self, desc: D, usage: D::Usage,
	) -> (ReadId<D::Resource>, WriteId<D::Resource>) {
		let real_id = self.frame.virtual_resources.len();
		let id = real_id.wrapping_add(self.frame.graph.resource_base_id);

		let ty = desc.ty(usage, self.frame.arena);

		self.frame.virtual_resources.push(VirtualResourceData {
			lifetime: ResourceLifetime::singular(self.frame.passes.len() as _),
			ty,
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

	/// Read CPU data that another pass outputs.
	pub fn data_input<T>(&mut self, id: &GetId<T>) { let _ = id; }

	/// Just like [`Self::data_input`], but the pass only gets a reference to the data.
	pub fn data_input_ref<T>(&mut self, id: RefId<T>) { let _ = id; }

	/// Output some CPU data for other passes.
	pub fn data_output<T>(&mut self) -> (SetId<T>, GetId<T>) {
		let real_id = self.frame.virtual_resources.len();
		let id = real_id.wrapping_add(self.frame.graph.resource_base_id);

		self.frame.virtual_resources.push(VirtualResourceData {
			lifetime: ResourceLifetime::singular(self.frame.passes.len() as _),
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
		};
		self.frame.passes.push(pass);
	}
}

pub struct PassContext<'frame, 'graph> {
	pub arena: &'graph Arena,
	pub device: &'frame Device,
	pub buf: CommandBuffer,
	base_id: usize,
	pass: u32,
	resource_map: &'frame mut ResourceMap<'graph>,
	caches: &'frame mut Caches,
}

impl<'frame, 'graph> PassContext<'frame, 'graph> {
	pub fn get_data_ref<T: 'frame>(&mut self, id: RefId<T>) -> &'frame T {
		let id = id.id.wrapping_sub(self.base_id);
		unsafe {
			let res = self.resource_map.get(id as u32);
			let (ptr, state) = res.data();

			assert!(
				matches!(state, DataState::Init { .. }),
				"Transient Data has not been initialized"
			);
			&*ptr.as_ptr()
		}
	}

	pub fn get_data<T: 'frame>(&mut self, id: GetId<T>) -> T {
		let id = id.id.wrapping_sub(self.base_id);
		unsafe {
			let res = self.resource_map.get(id as u32);
			let (ptr, state) = res.data::<T>();

			assert!(
				matches!(state, DataState::Init { .. }),
				"Transient Data has not been initialized"
			);
			let data = ptr.as_ptr().read();
			*state = DataState::Uninit;
			data
		}
	}

	pub fn set_data<T: 'frame>(&mut self, id: SetId<T>, data: T) {
		let id = id.id.wrapping_sub(self.base_id);
		unsafe {
			let res = self.resource_map.get(id as u32);
			let (ptr, state) = res.data::<T>();

			ptr.as_ptr().write(data);
			*state = DataState::Init {
				drop: |ptr| {
					let ptr = ptr.as_ptr() as *mut T;
					ptr.drop_in_place();
				},
			}
		}
	}

	pub fn read<T: VirtualResource>(&mut self, id: ReadId<T>) -> T {
		let id = id.id.wrapping_sub(self.base_id);
		unsafe {
			let res = self.resource_map.get(id as u32);
			T::from_res(self.pass, res, &mut self.caches, &self.device)
		}
	}

	pub fn write<T: VirtualResource>(&mut self, id: WriteId<T>) -> T {
		let id = id.id.wrapping_sub(self.base_id);
		unsafe {
			let res = self.resource_map.get(id as u32);
			T::from_res(self.pass, res, &mut self.caches, &self.device)
		}
	}
}

pub struct SetId<T> {
	id: usize,
	_marker: PhantomData<T>,
}

pub struct GetId<T> {
	id: usize,
	_marker: PhantomData<T>,
}

impl<T: Copy> Copy for GetId<T> {}
impl<T: Copy> Clone for GetId<T> {
	fn clone(&self) -> Self { *self }
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

struct PassData<'pass, 'graph> {
	// UTF-8 encoded, null terminated.
	name: Vec<u8, &'graph Arena>,
	callback: Box<dyn FnOnce(PassContext) + 'pass, &'graph Arena>,
}

type ArenaMap<'graph, K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>, &'graph Arena>;
