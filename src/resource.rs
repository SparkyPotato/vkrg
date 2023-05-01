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
		descriptor::{BufferId, ImageId, StorageImageId},
		Device,
		Queues,
	},
	Error,
	Result,
};

pub trait Resource: Default + Sized {
	type Desc: Eq + Hash + Copy;
	type Handle: Copy;

	fn handle(&self) -> Self::Handle;

	fn create(device: &Device, desc: Self::Desc) -> Result<Self>;

	unsafe fn destroy(self, device: &Device);
}

/// A description for a buffer.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct BufferDesc {
	pub size: usize,
	pub usage: BufferUsageFlags,
}

/// A GPU-side buffer.
#[derive(Default)]
struct Buffer {
	inner: ash::vk::Buffer,
	alloc: Allocation,
	id: Option<BufferId>,
}

impl Buffer {
	fn create(device: &Device, desc: BufferDesc, location: MemoryLocation) -> Result<Self> {
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

		let alloc = device
			.allocator()
			.allocate(&AllocationCreateDesc {
				name: "Graph Buffer",
				requirements: unsafe { device.device().get_buffer_memory_requirements(buffer) },
				location,
				linear: true,
			})
			.map_err(|e| Error::Message(e.to_string()))?;

		Ok(Self {
			inner: buffer,
			alloc,
			id,
		})
	}

	unsafe fn destroy(self, device: &Device) {
		if let Some(id) = self.id {
			device.base_descriptors().return_buffer(id);
		}

		let _ = device.allocator().free(self.alloc);
		device.device().destroy_buffer(self.inner, None);
	}
}

/// A handle to a buffer for uploading data from the CPU to the GPU.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct UploadBufferHandle {
	pub buffer: ash::vk::Buffer,
	pub id: Option<BufferId>,
	pub data: NonNull<[u8]>,
}

/// A buffer for uploading data from the CPU to the GPU.
#[derive(Default)]
pub struct UploadBuffer {
	inner: Buffer,
}

impl Resource for UploadBuffer {
	type Desc = BufferDesc;
	type Handle = UploadBufferHandle;

	fn handle(&self) -> Self::Handle {
		UploadBufferHandle {
			buffer: self.inner.inner,
			data: unsafe {
				NonNull::new_unchecked(std::ptr::slice_from_raw_parts_mut(
					self.inner.alloc.mapped_ptr().unwrap().as_ptr() as _,
					self.inner.alloc.size() as _,
				))
			},
			id: self.inner.id,
		}
	}

	fn create(device: &Device, desc: Self::Desc) -> Result<Self>
	where
		Self: Sized,
	{
		Buffer::create(device, desc, MemoryLocation::CpuToGpu).map(|inner| Self { inner })
	}

	unsafe fn destroy(self, device: &Device) { self.inner.destroy(device) }
}

/// A handle to a buffer on the GPU.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct GpuBufferHandle {
	pub buffer: ash::vk::Buffer,
	pub id: Option<BufferId>,
}

/// A buffer on the GPU.
#[derive(Default)]
pub struct GpuBuffer {
	inner: Buffer,
}

impl Resource for GpuBuffer {
	type Desc = BufferDesc;
	type Handle = GpuBufferHandle;

	fn handle(&self) -> Self::Handle {
		GpuBufferHandle {
			buffer: self.inner.inner,
			id: self.inner.id,
		}
	}

	fn create(device: &Device, desc: Self::Desc) -> Result<Self> {
		Buffer::create(device, desc, MemoryLocation::GpuOnly).map(|inner| Self { inner })
	}

	unsafe fn destroy(self, device: &Device) { self.inner.destroy(device) }
}

/// A description for an image.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct ImageDesc {
	pub flags: ImageCreateFlags,
	pub format: Format,
	pub size: Extent3D,
	pub levels: u32,
	pub layers: u32,
	pub samples: SampleCountFlags,
	pub usage: ImageUsageFlags,
}

/// A GPU-side image.
#[derive(Default)]
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
					.flags(desc.flags)
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
					.sharing_mode(SharingMode::EXCLUSIVE)
					.initial_layout(ImageLayout::UNDEFINED),
				None,
			)?
		};

		let alloc = device
			.allocator()
			.allocate(&AllocationCreateDesc {
				name: "Graph Image",
				requirements: unsafe { device.device().get_image_memory_requirements(image) },
				location: MemoryLocation::GpuOnly,
				linear: false,
			})
			.map_err(|e| Error::Message(e.to_string()))?;

		Ok(Self { inner: image, alloc })
	}

	unsafe fn destroy(self, device: &Device) {
		let _ = device.allocator().free(self.alloc);
		device.device().destroy_image(self.inner, None);
	}
}

/// The usage of an image view.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub enum ImageViewUsage {
	None,
	Sampled,
	Storage,
	Both,
}

/// A description for an image view.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct ImageViewDesc {
	pub image: ash::vk::Image,
	pub view_type: ImageViewType,
	pub format: Format,
	pub usage: ImageViewUsage,
}

/// A GPU-side image view.
#[derive(Default, Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct ImageView {
	pub image: ash::vk::Image,
	pub view: ash::vk::ImageView,
	pub id: Option<ImageId>,
	pub storage_id: Option<StorageImageId>,
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
						aspect_mask: image_aspect_mask(desc.format),
						base_mip_level: 0,
						level_count: REMAINING_MIP_LEVELS,
						base_array_layer: 0,
						layer_count: REMAINING_ARRAY_LAYERS,
					}),
				None,
			)?;
			let layout = sampled_image_layout(desc.format);
			let (id, storage_id) = match desc.usage {
				ImageViewUsage::None => (None, None),
				ImageViewUsage::Sampled => (
					Some(device.base_descriptors().get_image(device.device(), view, layout)),
					None,
				),
				ImageViewUsage::Storage => (
					None,
					Some(device.base_descriptors().get_storage_image(device.device(), view)),
				),
				ImageViewUsage::Both => (
					Some(device.base_descriptors().get_image(device.device(), view, layout)),
					Some(device.base_descriptors().get_storage_image(device.device(), view)),
				),
			};

			Ok(Self {
				image: desc.image,
				view,
				id,
				storage_id,
			})
		}
	}

	unsafe fn destroy(self, device: &Device) {
		unsafe {
			if let Some(id) = self.id {
				device.base_descriptors().return_image(id);
			}
			if let Some(id) = self.storage_id {
				device.base_descriptors().return_storage_image(id);
			}
			device.device().destroy_image_view(self.view, None);
		}
	}
}

pub(crate) fn image_aspect_mask(format: Format) -> ImageAspectFlags {
	match format {
		Format::D16_UNORM
		| Format::D32_SFLOAT
		| Format::D16_UNORM_S8_UINT
		| Format::D24_UNORM_S8_UINT
		| Format::D32_SFLOAT_S8_UINT => ImageAspectFlags::DEPTH | ImageAspectFlags::STENCIL,
		Format::S8_UINT => ImageAspectFlags::STENCIL,
		_ => ImageAspectFlags::COLOR,
	}
}

pub(crate) fn sampled_image_layout(format: Format) -> ImageLayout {
	match format {
		Format::D16_UNORM | Format::X8_D24_UNORM_PACK32 | Format::D32_SFLOAT => ImageLayout::DEPTH_READ_ONLY_OPTIMAL,
		Format::D16_UNORM_S8_UINT | Format::D24_UNORM_S8_UINT | Format::D32_SFLOAT_S8_UINT => {
			ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
		},
		Format::S8_UINT => ImageLayout::STENCIL_READ_ONLY_OPTIMAL,
		_ => ImageLayout::SHADER_READ_ONLY_OPTIMAL,
	}
}
