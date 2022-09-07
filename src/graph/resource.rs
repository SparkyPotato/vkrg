use std::{hash::Hash, ptr::NonNull};

use ash::vk::{
	BufferCreateInfo,
	BufferUsageFlags,
	ComponentMapping,
	ComponentSwizzle,
	Extent3D,
	Format,
	ImageAspectFlags,
	ImageCreateFlags,
	ImageCreateInfo,
	ImageLayout,
	ImageSubresourceRange,
	ImageType,
	ImageUsageFlags,
	ImageViewCreateInfo,
	ImageViewType,
	SampleCountFlags,
	SharingMode,
	REMAINING_ARRAY_LAYERS,
	REMAINING_MIP_LEVELS,
};
use gpu_allocator::{
	vulkan::{Allocation, AllocationCreateDesc},
	MemoryLocation,
};

use crate::{
	device::{
		descriptor::{BufferId, ImageId},
		Device,
		Queues,
	},
	Error,
	Result,
};

pub trait Resource {
	type Handle: Copy;
	type Desc: Eq + Hash + Copy;

	fn handle(&self) -> Self::Handle;

	fn create(device: &Device, desc: Self::Desc) -> Result<Self>
	where
		Self: Sized;

	unsafe fn destroy(&mut self, device: &Device);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UploadBufferHandle {
	pub buffer: ash::vk::Buffer,
	pub id: Option<BufferId>,
	pub data: NonNull<[u8]>,
}

#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub enum UploadBufferUsage {
	Shader,
	Index,
	Copy,
	Indirect,
}

#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct UploadBufferDesc {
	pub size: usize,
	pub usage: UploadBufferUsage,
}

pub struct UploadBuffer {
	inner: ash::vk::Buffer,
	alloc: Allocation,
	id: Option<BufferId>,
}

impl Resource for UploadBuffer {
	type Desc = UploadBufferDesc;
	type Handle = UploadBufferHandle;

	fn handle(&self) -> Self::Handle {
		UploadBufferHandle {
			buffer: self.inner,
			data: unsafe {
				NonNull::new_unchecked(std::ptr::slice_from_raw_parts_mut(
					self.alloc.mapped_ptr().unwrap().as_ptr() as _,
					self.alloc.size() as _,
				))
			},
			id: self.id,
		}
	}

	fn create(device: &Device, desc: Self::Desc) -> Result<Self> {
		let info = BufferCreateInfo::builder()
			.size(desc.size as u64)
			.usage(match desc.usage {
				UploadBufferUsage::Shader => BufferUsageFlags::STORAGE_BUFFER,
				UploadBufferUsage::Index => BufferUsageFlags::INDEX_BUFFER,
				UploadBufferUsage::Copy => BufferUsageFlags::TRANSFER_SRC,
				UploadBufferUsage::Indirect => BufferUsageFlags::INDIRECT_BUFFER,
			})
			.sharing_mode(SharingMode::CONCURRENT);

		let usage = info.usage;
		let buffer = unsafe {
			match device.queue_families() {
				Queues::Single(q) => device.device().create_buffer(&info.queue_family_indices(&[q]), None),
				Queues::Separate {
					graphics,
					compute,
					transfer,
				} => device
					.device()
					.create_buffer(&info.queue_family_indices(&[graphics, compute, transfer]), None),
			}
		}?;

		let id = usage
			.contains(BufferUsageFlags::STORAGE_BUFFER)
			.then(|| device.base_descriptors().get_buffer(device.device(), buffer));

		let requirements = unsafe { device.device().get_buffer_memory_requirements(buffer) };
		let alloc = device
			.allocator()
			.allocate(&AllocationCreateDesc {
				name: "CPU to GPU Graph Buffer",
				requirements,
				location: MemoryLocation::CpuToGpu,
				linear: true,
			})
			.map_err(|e| Error::Message(e.to_string()))?;

		Ok(Self {
			inner: buffer,
			alloc,
			id,
		})
	}

	unsafe fn destroy(&mut self, device: &Device) {
		let _ = device.allocator().free(std::mem::take(&mut self.alloc));
		device.device().destroy_buffer(self.inner, None);
	}
}

#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct GpuBufferHandle {
	pub buffer: ash::vk::Buffer,
	pub id: Option<BufferId>,
}

#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct GpuBufferDesc {
	pub size: usize,
	pub usage: BufferUsageFlags,
}

pub struct GpuBuffer {
	inner: ash::vk::Buffer,
	alloc: Allocation,
	id: Option<BufferId>,
}

impl Resource for GpuBuffer {
	type Desc = GpuBufferDesc;
	type Handle = GpuBufferHandle;

	fn handle(&self) -> Self::Handle {
		GpuBufferHandle {
			buffer: self.inner,
			id: self.id,
		}
	}

	fn create(device: &Device, desc: Self::Desc) -> Result<Self> {
		let info = BufferCreateInfo::builder()
			.size(desc.size as u64)
			.usage(desc.usage)
			.sharing_mode(SharingMode::CONCURRENT);

		let usage = info.usage;
		let buffer = unsafe {
			match device.queue_families() {
				Queues::Single(q) => device.device().create_buffer(&info.queue_family_indices(&[q]), None),
				Queues::Separate {
					graphics,
					compute,
					transfer,
				} => device
					.device()
					.create_buffer(&info.queue_family_indices(&[graphics, compute, transfer]), None),
			}
		}?;

		let id = usage
			.contains(BufferUsageFlags::STORAGE_BUFFER)
			.then(|| device.base_descriptors().get_buffer(device.device(), buffer));

		let requirements = unsafe { device.device().get_buffer_memory_requirements(buffer) };
		let alloc = device
			.allocator()
			.allocate(&AllocationCreateDesc {
				name: "CPU to GPU Graph Buffer",
				requirements,
				location: MemoryLocation::CpuToGpu,
				linear: true,
			})
			.map_err(|e| Error::Message(e.to_string()))?;

		Ok(Self {
			inner: buffer,
			alloc,
			id,
		})
	}

	unsafe fn destroy(&mut self, device: &Device) {
		let _ = device.allocator().free(std::mem::take(&mut self.alloc));
		device.device().destroy_buffer(self.inner, None);
	}
}

#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub enum ImageFlags {
	Cube,
	Array,
	CubeAndArray,
}

#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct ImageDesc {
	pub flags: ImageFlags,
	pub format: Format,
	pub size: Extent3D,
	pub levels: u32,
	pub layers: u32,
	pub samples: SampleCountFlags,
	pub usage: ImageUsageFlags,
}

pub struct Image {
	inner: ash::vk::Image,
	alloc: Allocation,
}

impl Resource for Image {
	type Desc = ImageDesc;
	type Handle = ash::vk::Image;

	fn handle(&self) -> Self::Handle { self.inner }

	fn create(device: &Device, desc: Self::Desc) -> Result<Self> {
		let image = unsafe {
			device.device().create_image(
				&ImageCreateInfo::builder()
					.flags(match desc.flags {
						ImageFlags::Cube => ImageCreateFlags::CUBE_COMPATIBLE,
						ImageFlags::Array => ImageCreateFlags::TYPE_2D_ARRAY_COMPATIBLE,
						ImageFlags::CubeAndArray => {
							ImageCreateFlags::CUBE_COMPATIBLE | ImageCreateFlags::TYPE_2D_ARRAY_COMPATIBLE
						},
					})
					.image_type(if desc.size.depth > 1 {
						ImageType::TYPE_3D
					} else if desc.size.height > 1 {
						ImageType::TYPE_2D
					} else {
						ImageType::TYPE_1D
					})
					.format(desc.format)
					.extent(desc.size)
					.mip_levels(desc.levels)
					.array_layers(desc.layers)
					.samples(desc.samples)
					.usage(desc.usage)
					.sharing_mode(SharingMode::CONCURRENT)
					.initial_layout(ImageLayout::UNDEFINED),
				None,
			)?
		};

		let alloc = device
			.allocator()
			.allocate(&AllocationCreateDesc {
				name: "CPU to GPU Graph Image",
				requirements: unsafe { device.device().get_image_memory_requirements(image) },
				location: MemoryLocation::CpuToGpu,
				linear: true,
			})
			.map_err(|e| Error::Message(e.to_string()))?;

		Ok(Self { inner: image, alloc })
	}

	unsafe fn destroy(&mut self, device: &Device) {
		let _ = device.allocator().free(std::mem::take(&mut self.alloc));
		device.device().destroy_image(self.inner, None);
	}
}

#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub enum ImageViewUsage {
	Sampled,
	Storage,
	Both,
}

#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct ImageViewDesc {
	pub image: ash::vk::Image,
	pub view_type: ImageViewType,
	pub format: Format,
	pub usage: ImageViewUsage,
}

#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct ImageView {
	inner: ash::vk::ImageView,
	id: Option<ImageId>,
	storage_id: Option<ImageId>,
}

impl Resource for ImageView {
	type Desc = ImageViewDesc;
	type Handle = Self;

	fn handle(&self) -> Self::Handle { *self }

	fn create(device: &Device, desc: Self::Desc) -> Result<Self> {
		unsafe {
			let view = device.device().create_image_view(
				&ImageViewCreateInfo::builder()
					.image(desc.image)
					.view_type(desc.view_type)
					.format(desc.format)
					.components(ComponentMapping {
						r: ComponentSwizzle::IDENTITY,
						g: ComponentSwizzle::IDENTITY,
						b: ComponentSwizzle::IDENTITY,
						a: ComponentSwizzle::IDENTITY,
					})
					.subresource_range(ImageSubresourceRange {
						aspect_mask: match desc.format {
							Format::D16_UNORM
							| Format::D32_SFLOAT
							| Format::D16_UNORM_S8_UINT
							| Format::D24_UNORM_S8_UINT
							| Format::D32_SFLOAT_S8_UINT => ImageAspectFlags::DEPTH,
							Format::S8_UINT => ImageAspectFlags::STENCIL,
							_ => ImageAspectFlags::COLOR,
						},
						base_mip_level: 0,
						level_count: REMAINING_MIP_LEVELS,
						base_array_layer: 0,
						layer_count: REMAINING_ARRAY_LAYERS,
					}),
				None,
			)?;
			let layout = match desc.format {
				Format::D16_UNORM
				| Format::D32_SFLOAT
				| Format::D16_UNORM_S8_UINT
				| Format::D24_UNORM_S8_UINT
				| Format::D32_SFLOAT_S8_UINT => ImageLayout::DEPTH_READ_ONLY_OPTIMAL,
				_ => ImageLayout::SHADER_READ_ONLY_OPTIMAL,
			};
			let (id, storage_id) = match desc.usage {
				ImageViewUsage::Sampled => (
					Some(device.base_descriptors().get_image(device.device(), view, layout)),
					None,
				),
				ImageViewUsage::Storage => (
					None,
					Some(
						device
							.base_descriptors()
							.get_image(device.device(), view, ImageLayout::GENERAL),
					),
				),
				ImageViewUsage::Both => (
					Some(device.base_descriptors().get_image(device.device(), view, layout)),
					Some(
						device
							.base_descriptors()
							.get_image(device.device(), view, ImageLayout::GENERAL),
					),
				),
			};

			Ok(Self {
				inner: view,
				id,
				storage_id,
			})
		}
	}

	unsafe fn destroy(&mut self, device: &Device) {
		unsafe {
			device.device().destroy_image_view(self.inner, None);
		}
	}
}
