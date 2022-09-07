use rustc_hash::FxHashMap;

use crate::{device::Device, graph::resource::Resource, Result};

/// A resource that has its usage generations tracked.
struct TrackedResource<T: Resource> {
	inner: T,
	/// Number of generations the resource has been unused for.
	unused: u8,
}

/// A list of resources with the same descriptor.
struct ResourceList<T: Resource> {
	cursor: usize,
	// Stored in most recently used order.
	resources: Vec<TrackedResource<T>>,
}

impl<T: Resource> ResourceList<T> {
	const DESTROY_LAG: u8 = 2;

	pub fn new() -> Self {
		Self {
			cursor: 0,
			resources: Vec::new(),
		}
	}

	pub fn get_or_create(&mut self, device: &Device, desc: T::Desc) -> Result<T::Handle> {
		let ret = match self.resources.get_mut(self.cursor) {
			Some(resource) => {
				resource.unused = 0;
				resource.inner.handle()
			},
			None => {
				let resource = T::create(device, desc)?;
				let handle = resource.handle();
				self.resources.push(TrackedResource {
					inner: resource,
					unused: 0,
				});
				handle
			},
		};
		self.cursor += 1;
		Ok(ret)
	}

	pub unsafe fn reset(&mut self, device: &Device) {
		// Everything before the cursor was just used.
		let mut first_destroyable = self.cursor;

		for resource in self.resources[self.cursor..].iter_mut() {
			// Everything after this has not been used for at least `DESTROY_LAG` generations.
			resource.unused += 1;
			if resource.unused >= Self::DESTROY_LAG {
				break;
			}
			first_destroyable += 1;
		}
		for mut resource in self.resources.drain(first_destroyable..).rev() {
			resource.inner.destroy(device);
		}
		self.cursor = 0;
	}
}

pub struct ResourceCache<T: Resource> {
	resources: FxHashMap<T::Desc, ResourceList<T>>,
}

impl<T: Resource> ResourceCache<T> {
	/// Create an empty cache.
	pub fn new() -> Self {
		Self {
			resources: FxHashMap::default(),
		}
	}

	/// Reset the cache, incrementing the generation.
	///
	/// # Safety
	/// All resources returned by [`Self::get`] must not be used after this call.
	pub unsafe fn reset(&mut self, device: &Device) {
		for list in self.resources.iter_mut() {
			list.1.reset(device);
		}
	}

	/// Get an unused resource with the given descriptor. Is valid until [`Self::reset`] is called.
	pub fn get(&mut self, device: &Device, desc: T::Desc) -> Result<T::Handle> {
		let list = self.resources.entry(desc).or_insert_with(ResourceList::new);
		list.get_or_create(device, desc)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::Result;

	#[test]
	fn resource_list() {
		#[derive(Copy, Clone)]
		struct Resource;
		#[derive(Copy, Clone, PartialEq, Eq, Hash)]
		struct ResourceDesc;

		impl super::Resource for Resource {
			type Desc = ResourceDesc;
			type Handle = Self;

			fn handle(&self) -> Self::Handle { *self }

			fn create(_: &Device, _: Self::Desc) -> Result<Self> { Ok(Resource) }

			unsafe fn destroy(&self, _: &Device) {}
		}

		let device = Device::new().unwrap();
		let mut list = ResourceList::<Resource>::new();

		list.get_or_create(&device, ResourceDesc).unwrap();
		list.get_or_create(&device, ResourceDesc).unwrap();
		list.get_or_create(&device, ResourceDesc).unwrap();

		assert_eq!(list.resources.len(), 3);
		unsafe {
			list.reset(&device);
		}

		list.get_or_create(&device, ResourceDesc).unwrap();

		assert_eq!(list.resources.len(), 3);
		unsafe {
			list.reset(&device);
		}

		list.get_or_create(&device, ResourceDesc).unwrap();

		assert_eq!(list.resources.len(), 3);
		unsafe {
			list.reset(&device);
		}

		assert_eq!(list.resources.len(), 1);
	}
}
