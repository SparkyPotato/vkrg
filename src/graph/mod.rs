use std::{
	alloc::{Allocator, Layout},
	marker::PhantomData,
	ptr::NonNull,
};

use crate::{
	arena::{collect_allocated_vec, Arena},
	graph::{
		cache::ResourceCache,
		resource::{GpuBuffer, Image, ImageView, UploadBuffer},
	},
};

pub mod cache;
pub mod resource;
#[cfg(test)]
mod test;

/// The render graph.
pub struct RenderGraph {
	caches: Caches,
	curr_frame: usize,
	transient_base_id: usize,
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
			transient_base_id: 0,
		}
	}

	pub fn frame<'pass, 'graph>(&'graph mut self, arena: &'graph Arena) -> Frame<'pass, 'graph> {
		Frame {
			graph: self,
			arena,
			passes: Vec::new_in(arena),
			transient_data: Vec::new_in(arena),
		}
	}
}

pub struct Frame<'pass, 'graph> {
	graph: &'graph mut RenderGraph,
	arena: &'graph Arena,
	passes: Vec<PassData<'pass, 'graph>, &'graph Arena>,
	transient_data: Vec<NonNull<()>, &'graph Arena>,
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
			(pass.callback)(PassContext { frame: &self });
		}

		self.graph.curr_frame ^= 1;
		self.graph.transient_base_id = self.graph.transient_base_id.wrapping_add(self.transient_data.len());
	}
}

pub struct PassBuilder<'frame, 'pass, 'graph> {
	name: Vec<u8, &'graph Arena>,
	frame: &'frame mut Frame<'pass, 'graph>,
	inputs: Vec<DependencyType, &'graph Arena>,
	outputs: Vec<DependencyType, &'graph Arena>,
}

impl<'frame, 'pass, 'graph> PassBuilder<'frame, 'pass, 'graph> {
	/// Create an input from another pass' output.
	pub fn input(&mut self) {}

	/// Create an pass-local transient resource. This is like an output, but it can't be used as an input.
	pub fn inner(&mut self) {}

	/// This pass outputs some GPU-side data for other passes.
	pub fn output(&mut self) {}

	/// Create an input from another pass' CPU-side data.
	pub fn data_input<T>(&mut self, id: &GetId<T>) {
		self.inputs.push(DependencyType::Data(
			id.id.wrapping_sub(self.frame.graph.transient_base_id),
		));
	}

	/// Just like [`Self::data_input`], but the pass only gets a reference to the data.
	pub fn data_input_ref<T>(&mut self, id: RefId<T>) {
		self.inputs.push(DependencyType::Data(
			id.id.wrapping_sub(self.frame.graph.transient_base_id),
		));
	}

	/// This pass outputs some CPU-side data for other passes.
	pub fn data_output<T>(&mut self) -> (SetId<T>, GetId<T>) {
		let real_id = self.frame.transient_data.len();
		let id = real_id.wrapping_add(self.frame.graph.transient_base_id);
		self.outputs.push(DependencyType::Data(real_id));
		self.frame
			.transient_data
			.push(self.frame.arena.allocate(Layout::new::<T>()).unwrap().cast());
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

pub struct SetId<T> {
	id: usize,
	_marker: PhantomData<T>,
}

impl<T> Drop for SetId<T> {
	fn drop(&mut self) {
		panic!(
			"dropped SetId<{}> without using it to initialize output",
			std::any::type_name::<T>()
		);
	}
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

pub struct PassContext<'frame, 'pass, 'graph> {
	frame: &'frame Frame<'pass, 'graph>,
}

impl<'frame, 'pass, 'graph> PassContext<'frame, 'pass, 'graph> {
	pub fn arena(&self) -> &'graph Arena { self.frame.arena }

	pub fn get_data_ref<T: 'frame>(&self, id: RefId<T>) -> &'frame T {
		let id = id.id.wrapping_sub(self.frame.graph.transient_base_id);
		assert!(id < self.frame.transient_data.len(), "RefId from previous frame used");
		unsafe { &*self.frame.transient_data[id].cast::<T>().as_ptr() }
	}

	pub fn get_data<T: 'frame>(&self, id: GetId<T>) -> T {
		let id = id.id.wrapping_sub(self.frame.graph.transient_base_id);
		assert!(id < self.frame.transient_data.len(), "GetId from previous frame used");
		unsafe { self.frame.transient_data[id].cast::<T>().as_ptr().read() }
	}

	pub fn set_data<T: 'frame>(&mut self, id: SetId<T>, data: T) {
		let i = id.id;
		std::mem::forget(id); // Defuse drop bomb.
		let id = i.wrapping_sub(self.frame.graph.transient_base_id);
		assert!(id < self.frame.transient_data.len(), "SetId from previous frame used");
		let pointer = self.frame.transient_data[id];
		unsafe { pointer.as_ptr().cast::<T>().write(data) };
	}
}

struct PassData<'pass, 'graph> {
	// UTF-8 encoded, null terminated.
	name: Vec<u8, &'graph Arena>,
	callback: Box<dyn for<'frame> FnOnce(PassContext<'frame, 'pass, 'graph>) + 'pass, &'graph Arena>,
	inputs: Vec<DependencyType, &'graph Arena>,
	outputs: Vec<DependencyType, &'graph Arena>,
}

enum DependencyType {
	Data(usize),
	Resource(usize),
}
