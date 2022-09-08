use std::{
	alloc::{Allocator, Layout},
	hint::unreachable_unchecked,
	marker::PhantomData,
	ptr::NonNull,
};

use tracing::{span, Level};

pub use crate::graph::resource::{GpuBufferHandle, ImageView, UploadBufferHandle};
use crate::{
	arena::{collect_allocated_vec, Arena},
	graph::{
		cache::ResourceCache,
		resource::{GpuBuffer, Image, UploadBuffer},
	},
};

pub mod cache;
mod compile;
pub mod resource;
#[cfg(test)]
mod test;

/// The render graph.
pub struct RenderGraph {
	caches: Caches,
	curr_frame: usize,
	resource_base_id: usize,
}

struct Caches {
	upload_buffers: [ResourceCache<UploadBuffer>; 2],
	gpu_buffers: ResourceCache<GpuBuffer>,
	images: ResourceCache<Image>,
	image_views: ResourceCache<ImageView>,
}

impl RenderGraph {
	pub fn new() -> Self {
		let caches = Caches {
			upload_buffers: [ResourceCache::new(), ResourceCache::new()],
			gpu_buffers: ResourceCache::new(),
			images: ResourceCache::new(),
			image_views: ResourceCache::new(),
		};

		Self {
			caches,
			curr_frame: 0,
			resource_base_id: 0,
		}
	}

	pub fn frame<'pass, 'graph>(&'graph mut self, arena: &'graph Arena) -> Frame<'pass, 'graph> {
		Frame {
			graph: self,
			arena,
			passes: Vec::new_in(arena),
			transient_resources: Vec::new_in(arena),
		}
	}
}

pub struct Frame<'pass, 'graph> {
	graph: &'graph mut RenderGraph,
	arena: &'graph Arena,
	passes: Vec<PassData<'pass, 'graph>, &'graph Arena>,
	transient_resources: Vec<TransientResource, &'graph Arena>,
}

impl<'pass, 'graph> Frame<'pass, 'graph> {
	pub fn pass<'frame>(&'frame mut self, name: &str) -> PassBuilder<'frame, 'pass, 'graph> {
		let arena = self.arena;
		let name = name.as_bytes().iter().copied().chain([0]);
		PassBuilder {
			name: collect_allocated_vec(name, arena),
			frame: self,
			inputs: Vec::new_in(arena),
			outputs: Vec::new_in(arena),
		}
	}

	pub fn run(mut self) {
		let passes = std::mem::replace(&mut self.passes, Vec::new_in(self.arena));

		// We have ensured that passes are inserted in topological order during building.
		for pass in passes {
			{
				let span = span!(
					Level::TRACE,
					"run pass",
					name = unsafe { &std::str::from_utf8_unchecked(&pass.name[..pass.name.len()]) }
				);
				let _e = span.enter();

				(pass.callback)(PassContext { frame: &mut self });
			}
		}

		self.graph.curr_frame ^= 1;
		self.graph.resource_base_id = self.graph.resource_base_id.wrapping_add(self.transient_resources.len());
	}
}

pub struct PassBuilder<'frame, 'pass, 'graph> {
	name: Vec<u8, &'graph Arena>,
	frame: &'frame mut Frame<'pass, 'graph>,
	inputs: Vec<Dependency, &'graph Arena>,
	outputs: Vec<Dependency, &'graph Arena>,
}

impl<'frame, 'pass, 'graph> PassBuilder<'frame, 'pass, 'graph> {
	/// Create an input from another pass' output.
	pub fn input<T: VirtualResource>(&mut self, id: ReadId<T>) {
		self.inputs
			.push(Dependency(id.id.wrapping_sub(self.frame.graph.resource_base_id)));
	}

	/// Create an pass-local transient resource. This is like an output, but it can't be used as an input of another
	/// pass.
	pub fn inner<D: VirtualResourceDesc>(&mut self, desc: D) -> InnerId<D::Resource> {
		let real_id = self.frame.transient_resources.len();
		let id = real_id.wrapping_add(self.frame.graph.resource_base_id);

		self.frame.transient_resources.push(TransientResource {
			ty: D::ty(),
			pass: self.frame.passes.len(),
		});

		InnerId {
			id,
			_marker: PhantomData,
		}
	}

	/// This pass outputs some GPU data for other passes.
	pub fn output<D: VirtualResourceDesc>(&mut self, desc: D) -> (ReadId<D::Resource>, WriteId<D::Resource>) {
		let real_id = self.frame.transient_resources.len();
		let id = real_id.wrapping_add(self.frame.graph.resource_base_id);

		self.outputs.push(Dependency(real_id));
		self.frame.transient_resources.push(TransientResource {
			ty: D::ty(),
			pass: self.frame.passes.len(),
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
	pub fn data_input<T>(&mut self, id: &GetId<T>) {
		self.inputs
			.push(Dependency(id.id.wrapping_sub(self.frame.graph.resource_base_id)));
	}

	/// Just like [`Self::data_input`], but the pass only gets a reference to the data.
	pub fn data_input_ref<T>(&mut self, id: RefId<T>) {
		self.inputs
			.push(Dependency(id.id.wrapping_sub(self.frame.graph.resource_base_id)));
	}

	/// This pass outputs some CPU data for other passes.
	pub fn data_output<T>(&mut self) -> (SetId<T>, GetId<T>) {
		let real_id = self.frame.transient_resources.len();
		let id = real_id.wrapping_add(self.frame.graph.resource_base_id);

		self.outputs.push(Dependency(real_id));
		self.frame.transient_resources.push(TransientResource {
			pass: self.frame.passes.len(),
			ty: TransientResourceType::Data(self.frame.arena.allocate(Layout::new::<T>()).unwrap().cast(), false),
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

	pub fn build(self, callback: impl for<'f> FnOnce(PassContext<'f, 'pass, 'graph>) + 'pass) {
		let pass = PassData {
			name: self.name,
			callback: Box::new_in(callback, self.frame.arena),
			inputs: self.inputs,
			outputs: self.outputs,
		};
		self.frame.passes.push(pass);
	}
}

pub struct PassContext<'frame, 'pass, 'graph> {
	frame: &'frame mut Frame<'pass, 'graph>,
}

impl<'frame, 'pass, 'graph> PassContext<'frame, 'pass, 'graph> {
	pub fn arena(&self) -> &'graph Arena { self.frame.arena }

	pub fn get_data_ref<T: 'frame>(&self, id: RefId<T>) -> &'frame T {
		let id = id.id.wrapping_sub(self.frame.graph.resource_base_id);
		assert!(
			id < self.frame.transient_resources.len(),
			"RefId from previous frame used"
		);
		unsafe {
			let ty = &self.frame.transient_resources.get_unchecked(id).ty;

			assert!(ty.to_data::<T>().1, "Transient Data has not been initialized");
			&*ty.to_data::<T>().0.as_ptr()
		}
	}

	pub fn get_data<T: 'frame>(&self, id: GetId<T>) -> T {
		let id = id.id.wrapping_sub(self.frame.graph.resource_base_id);
		assert!(
			id < self.frame.transient_resources.len(),
			"GetId from previous frame used"
		);
		unsafe {
			let ty = &self.frame.transient_resources.get_unchecked(id).ty;

			assert!(ty.to_data::<T>().1, "Transient Data has not been initialized");
			ty.to_data::<T>().0.as_ptr().read()
		}
	}

	pub fn set_data<T: 'frame>(&mut self, id: SetId<T>, data: T) {
		let id = id.id.wrapping_sub(self.frame.graph.resource_base_id);
		unsafe {
			let ty = &mut self.frame.transient_resources.get_unchecked_mut(id).ty;

			ty.to_data::<T>().0.as_ptr().write(data);
			ty.init();
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

pub struct InnerId<T: VirtualResource> {
	id: usize,
	_marker: PhantomData<T>,
}

struct PassData<'pass, 'graph> {
	// UTF-8 encoded, null terminated.
	name: Vec<u8, &'graph Arena>,
	callback: Box<dyn for<'frame> FnOnce(PassContext<'frame, 'pass, 'graph>) + 'pass, &'graph Arena>,
	inputs: Vec<Dependency, &'graph Arena>,
	outputs: Vec<Dependency, &'graph Arena>,
}

struct Dependency(usize);

struct TransientResource {
	pass: usize,
	ty: TransientResourceType,
}

pub enum TransientResourceType {
	Data(NonNull<()>, bool),
	UploadBuffer,
	GpuBuffer,
	Image,
}

impl TransientResourceType {
	unsafe fn to_data<T>(&self) -> (NonNull<T>, bool) {
		match self {
			TransientResourceType::Data(ptr, b) => (ptr.cast(), *b),
			_ => unreachable_unchecked(),
		}
	}

	unsafe fn init(&mut self) {
		match self {
			TransientResourceType::Data(_, b) => {
				*b = true;
			},
			_ => unreachable_unchecked(),
		}
	}
}

/// This trait is sealed.
pub trait VirtualResourceDesc: sealed::Sealed {
	type Resource: VirtualResource;

	fn ty() -> TransientResourceType;
}

/// This trait is sealed.
pub trait VirtualResource: sealed::Sealed {
	type Desc: VirtualResourceDesc;
}

mod sealed {
	pub trait Sealed {}
}
