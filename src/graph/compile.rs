use std::{
	hash::{BuildHasherDefault, Hash, Hasher},
	hint::unreachable_unchecked,
	ops::RangeInclusive,
	ptr::NonNull,
};

use ash::vk::{Image, ImageMemoryBarrier2, MemoryBarrier2};
use hashbrown::HashMap;
use rustc_hash::FxHasher;
use tracing::{span, Level};

use crate::{
	arena::{collect_allocated_vec, Arena},
	device::Device,
	graph::{
		resource::{GpuBufferDesc, ImageDesc},
		Caches,
		Frame,
		GpuBufferHandle,
		InnerUsage,
		PassContext,
		ReadUsage,
		RenderGraph,
		UploadBufferDesc,
		UploadBufferHandle,
		VirtualResourceData,
		VirtualResourceType,
	},
};

pub struct CompiledFrame<'pass, 'graph> {
	pub passes: Vec<Pass<'pass, 'graph>, &'graph Arena>,
	pub resource_map: Vec<usize, &'graph Arena>,
	pub resources: Vec<Resource<'graph>, &'graph Arena>,
	pub graph: &'graph mut RenderGraph,
}

pub struct Pass<'pass, 'graph> {
	/// Null-terminated UTF-8 data.
	pub name: Vec<u8, &'graph Arena>,
	pub sync: Synchronization<'graph>,
	pub exec: Box<dyn FnOnce(PassContext) + 'pass, &'graph Arena>,
}

pub struct Synchronization<'graph> {
	pub barriers: Vec<MemoryBarrier2, &'graph Arena>,
	pub image_barriers: Vec<ImageMemoryBarrier2, &'graph Arena>,
}

pub enum Resource<'graph> {
	Data(NonNull<()>, bool),
	UploadBuffer(UploadBufferHandle),
	GpuBuffer(GpuBufferHandle),
	Image(Image, Vec<ReadUsage<InnerUsage>, &'graph Arena>),
}

impl<'graph> Resource<'graph> {
	pub unsafe fn data<T>(&self) -> (NonNull<T>, bool) {
		match self {
			Resource::Data(ptr, b) => (ptr.cast(), *b),
			_ => unreachable_unchecked(),
		}
	}

	pub unsafe fn init(&mut self) {
		match self {
			Resource::Data(_, b) => *b = true,
			_ => unreachable_unchecked(),
		}
	}

	pub unsafe fn upload_buffer(&self) -> UploadBufferHandle {
		match self {
			Resource::UploadBuffer(h) => *h,
			_ => unreachable_unchecked(),
		}
	}

	pub unsafe fn gpu_buffer(&self) -> GpuBufferHandle {
		match self {
			Resource::GpuBuffer(h) => *h,
			_ => unreachable_unchecked(),
		}
	}

	pub unsafe fn image(&self) -> (Image, &Vec<ReadUsage<InnerUsage>, &'graph Arena>) {
		match self {
			Resource::Image(i, u) => (*i, u),
			_ => unreachable_unchecked(),
		}
	}

	pub fn extend_with_desc(&mut self, desc: &mut PhysicalResourceDesc<'graph>) {
		match (self, desc) {
			(Self::Image(_, us), PhysicalResourceDesc::Image(_, them)) => us.extend(them.iter().copied()),
			_ => {},
		}
	}
}

impl<'pass, 'graph> Frame<'pass, 'graph> {
	pub fn compile(self, device: &Device) -> CompiledFrame<'pass, 'graph> {
		let span = span!(Level::TRACE, "compile graph");
		let _e = span.enter();

		// The order of the passes is already topologically sorted.
		// This is order we will run them in.

		let mut aliaser = ResourceAliaser::new(self.arena);
		let resource_map = collect_allocated_vec(
			self.virtual_resources
				.into_iter()
				.map(|res| aliaser.get_physical_resource(res, device, &mut self.graph.caches, self.graph.curr_frame)),
			self.arena,
		);

		CompiledFrame {
			passes: Vec::new_in(self.arena),
			resource_map,
			resources: aliaser.physical_resources,
			graph: self.graph,
		}
	}
}

struct ResourceAliaser<'graph> {
	physical_resources: Vec<Resource<'graph>, &'graph Arena>,
	lifetime_map: ArenaMap<'graph, PhysicalResourceDesc<'graph>, Vec<ResourceLifetime, &'graph Arena>>,
}

struct ResourceLifetime {
	lifetime: RangeInclusive<usize>,
	index: usize,
}

impl<'graph> ResourceAliaser<'graph> {
	fn new(arena: &'graph Arena) -> Self {
		Self {
			physical_resources: Vec::new_in(arena),
			lifetime_map: ArenaMap::with_hasher_in(BuildHasherDefault::default(), arena),
		}
	}

	fn get_physical_resource(
		&mut self, desc: VirtualResourceData<'graph>, device: &Device, caches: &mut Caches, frame: usize,
	) -> usize {
		let lifetime = desc.lifetime();
		let desc = PhysicalResourceDesc::from_virtual(desc.ty);

		let resources = self
			.lifetime_map
			.entry(desc.clone())
			.or_insert_with(|| Vec::new_in(self.physical_resources.allocator()));
		for res in resources.iter_mut() {
			if independent(res.lifetime.clone(), lifetime.clone()) {
				// We can reuse this resource, so extend its lifetime beyond us.
				res.lifetime = union(res.lifetime.clone(), lifetime.clone());
				return res.index;
			}
		}

		// We need to create a new resource.
		let index = self.physical_resources.len();
		resources.push(ResourceLifetime { lifetime, index });
		self.physical_resources.push(desc.to_res(device, caches, frame));
		index
	}
}

type ArenaMap<'graph, K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>, &'graph Arena>;

fn independent(lifetime: RangeInclusive<usize>, other: RangeInclusive<usize>) -> bool {
	lifetime.end() < other.start() || lifetime.start() > other.end()
}

fn union(lifetime: RangeInclusive<usize>, other: RangeInclusive<usize>) -> RangeInclusive<usize> {
	*lifetime.start().min(other.start())..=*lifetime.end().max(other.end())
}

#[doc(hidden)]
#[derive(Clone, Debug)]
pub enum PhysicalResourceDesc<'graph> {
	Data(NonNull<()>),
	UploadBuffer(UploadBufferDesc),
	GpuBuffer(GpuBufferDesc),
	Image(ImageDesc, Vec<ReadUsage<InnerUsage>, &'graph Arena>),
}

impl Hash for PhysicalResourceDesc<'_> {
	fn hash<H: Hasher>(&self, state: &mut H) {
		match self {
			PhysicalResourceDesc::Data(ptr) => {
				ptr.as_ptr().hash(state);
			},
			PhysicalResourceDesc::UploadBuffer(desc) => {
				desc.hash(state);
			},
			PhysicalResourceDesc::GpuBuffer(desc) => {
				desc.hash(state);
			},
			PhysicalResourceDesc::Image(desc, _) => {
				desc.hash(state);
			},
		}
	}
}

impl PartialEq for PhysicalResourceDesc<'_> {
	fn eq(&self, other: &Self) -> bool {
		match (self, other) {
			(PhysicalResourceDesc::Data(a), PhysicalResourceDesc::Data(b)) => a == b,
			(PhysicalResourceDesc::UploadBuffer(a), PhysicalResourceDesc::UploadBuffer(b)) => a == b,
			(PhysicalResourceDesc::GpuBuffer(a), PhysicalResourceDesc::GpuBuffer(b)) => a == b,
			(PhysicalResourceDesc::Image(a, _), PhysicalResourceDesc::Image(b, _)) => a == b,
			_ => false,
		}
	}
}

impl Eq for PhysicalResourceDesc<'_> {}

impl<'graph> PhysicalResourceDesc<'graph> {
	fn from_virtual(desc: VirtualResourceType<'graph>) -> Self {
		match desc {
			VirtualResourceType::Data(ptr) => Self::Data(ptr),
			VirtualResourceType::UploadBuffer(desc) => Self::UploadBuffer(desc),
			VirtualResourceType::GpuBuffer { desc, .. } => Self::GpuBuffer(desc),
			VirtualResourceType::Image { desc, read_usages, .. } => Self::Image(desc, read_usages),
		}
	}

	fn to_res(self, device: &Device, caches: &mut Caches, frame: usize) -> Resource<'graph> {
		match self {
			Self::Data(ptr) => Resource::Data(ptr, false),
			Self::UploadBuffer(desc) => Resource::UploadBuffer(
				caches.upload_buffers[frame]
					.get(device, desc)
					.expect("failed to create upload buffer"),
			),
			Self::GpuBuffer(desc) => Resource::GpuBuffer(
				caches
					.gpu_buffers
					.get(device, desc)
					.expect("failed to create gpu buffer"),
			),
			Self::Image(desc, read_usages) => Resource::Image(
				caches.images.get(device, desc).expect("failed to create image"),
				read_usages,
			),
		}
	}
}

#[cfg(test)]
mod test {
	use super::*;

	#[test]
	fn test_independent() {
		assert!(independent(0..=1, 2..=3));
		assert!(independent(2..=3, 0..=1));
		assert!(!independent(0..=1, 0..=1));
		assert!(!independent(0..=1, 0..=2));
		assert!(!independent(0..=2, 0..=1));
		assert!(!independent(0..=2, 1..=2));
		assert!(!independent(1..=2, 0..=2));
	}

	#[test]
	fn test_union() {
		assert_eq!(union(0..=1, 2..=3), 0..=3);
		assert_eq!(union(2..=3, 0..=1), 0..=3);
		assert_eq!(union(0..=1, 0..=1), 0..=1);
		assert_eq!(union(0..=1, 0..=2), 0..=2);
		assert_eq!(union(0..=2, 0..=1), 0..=2);
		assert_eq!(union(0..=2, 1..=2), 0..=2);
		assert_eq!(union(1..=2, 0..=2), 0..=2);
	}
}
