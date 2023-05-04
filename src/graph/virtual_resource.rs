use std::{hash::BuildHasherDefault, hint::unreachable_unchecked, ptr::NonNull};

use ash::vk::{
	AccessFlags2,
	BufferUsageFlags,
	Extent3D,
	Format,
	ImageCreateFlags,
	ImageLayout,
	ImageUsageFlags,
	ImageViewType,
	PipelineStageFlags2,
	SampleCountFlags,
	Semaphore,
};

use crate::{
	arena::Arena,
	device::Device,
	graph::{compile::Resource, ArenaMap, Caches},
	resource::{GpuBufferHandle, ImageView, ImageViewDesc, ImageViewUsage, UploadBufferHandle},
};

/// The usage of a buffer.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub enum BufferUsage {
	TransferSrc,
	TransferDst,
	Storage(Shader),
	Index,
	Vertex,
	Indirect,
}

impl From<BufferUsage> for BufferUsageFlags {
	fn from(usage: BufferUsage) -> Self {
		match usage {
			BufferUsage::TransferSrc => BufferUsageFlags::TRANSFER_SRC,
			BufferUsage::TransferDst => BufferUsageFlags::TRANSFER_DST,
			BufferUsage::Storage(_) => BufferUsageFlags::STORAGE_BUFFER,
			BufferUsage::Index => BufferUsageFlags::INDEX_BUFFER,
			BufferUsage::Vertex => BufferUsageFlags::VERTEX_BUFFER,
			BufferUsage::Indirect => BufferUsageFlags::INDIRECT_BUFFER,
		}
	}
}

/// A description for a buffer for uploading data from the CPU to the GPU.
///
/// Has a corresponding usage of [`BufferUsage`].
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct UploadBufferDesc {
	pub size: usize,
}

/// A description for a GPU buffer.
///
/// Has a corresponding usage of [`BufferUsage`].
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct GpuBufferDesc {
	pub size: usize,
}

/// A description for an image.
///
/// Has a corresponding usage of [`ImageUsage`].
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct ImageDesc {
	pub size: Extent3D,
	pub levels: u32,
	pub layers: u32,
	pub samples: SampleCountFlags,
}

/// A shader stage
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub enum Shader {
	Vertex,
	TesselationControl,
	TesselationEvaluation,
	Geometry,
	Fragment,
	Compute,
}

impl From<Shader> for PipelineStageFlags2 {
	fn from(shader: Shader) -> Self {
		match shader {
			Shader::Vertex => PipelineStageFlags2::VERTEX_SHADER,
			Shader::TesselationControl => PipelineStageFlags2::TESSELLATION_CONTROL_SHADER,
			Shader::TesselationEvaluation => PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER,
			Shader::Geometry => PipelineStageFlags2::GEOMETRY_SHADER,
			Shader::Fragment => PipelineStageFlags2::FRAGMENT_SHADER,
			Shader::Compute => PipelineStageFlags2::COMPUTE_SHADER,
		}
	}
}

/// The type of usage of an image.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub enum ImageUsageType {
	TransferSrc,
	TransferDst,
	Sampled(Shader),
	Storage(Shader),
	ColorAttachment,
	DepthStencilAttachment,
}

impl From<ImageUsageType> for ImageUsageFlags {
	fn from(usage: ImageUsageType) -> Self {
		match usage {
			ImageUsageType::TransferSrc => ImageUsageFlags::TRANSFER_SRC,
			ImageUsageType::TransferDst => ImageUsageFlags::TRANSFER_DST,
			ImageUsageType::Sampled(_) => ImageUsageFlags::SAMPLED,
			ImageUsageType::Storage(_) => ImageUsageFlags::STORAGE,
			ImageUsageType::ColorAttachment => ImageUsageFlags::COLOR_ATTACHMENT,
			ImageUsageType::DepthStencilAttachment => ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
		}
	}
}

/// The usage of an image.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct ImageUsage {
	pub format: Format,
	pub usage: ImageUsageType,
	pub view_type: ImageViewType,
}

impl ImageUsage {
	pub fn create_flags(&self) -> ImageCreateFlags {
		match self.view_type {
			ImageViewType::CUBE | ImageViewType::CUBE_ARRAY => ImageCreateFlags::CUBE_COMPATIBLE,
			ImageViewType::TYPE_2D_ARRAY => ImageCreateFlags::TYPE_2D_ARRAY_COMPATIBLE,
			_ => ImageCreateFlags::empty(),
		}
	}
}

/// Synchronization required to access an external resource.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, Default)]
pub struct ExternalSync<A> {
	/// The semaphore to wait on. If no cross-queue sync is required, this is `::null()`.
	pub semaphore: Semaphore,
	/// If `semaphore` is a timeline semaphore, the value to set.
	pub value: u64,
	/// The access to the resource.
	pub access: A,
}

/// An access of a resource.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Default)]
pub struct Access {
	pub stage: PipelineStageFlags2,
	pub access: AccessFlags2,
}

/// An access of an image.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Default)]
pub struct ImageAccess {
	pub access: Access,
	pub layout: ImageLayout,
}

/// A buffer external to the render graph.
///
/// Has a corresponding usage of [`BufferUsage`].
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct ExternalBuffer {
	pub handle: GpuBufferHandle,
	pub pre_sync: ExternalSync<Access>,
	pub post_sync: ExternalSync<Access>,
}

/// An image external to the render graph.
///
/// Has a corresponding usage of [`ImageUsage`].
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct ExternalImage {
	pub handle: ash::vk::Image,
	pub pre_sync: ExternalSync<ImageAccess>,
	pub post_sync: ExternalSync<Access>,
}

#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub struct ResourceLifetime {
	pub start: u32,
	pub end: u32,
}

impl ResourceLifetime {
	pub fn singular(pass: u32) -> Self { Self { start: pass, end: pass } }

	pub fn union(self, other: Self) -> Self {
		Self {
			start: self.start.min(other.start),
			end: self.end.max(other.end),
		}
	}

	pub fn independent(self, other: Self) -> bool { self.start > other.end || self.end < other.start }
}

pub struct VirtualResourceData<'graph> {
	pub lifetime: ResourceLifetime,
	pub ty: VirtualResourceType<'graph>,
}

pub trait VirtualResourceDesc {
	type Resource: VirtualResource;

	fn ty(self, write_usage: <Self::Resource as VirtualResource>::Usage, arena: &Arena) -> VirtualResourceType;
}

pub trait VirtualResource {
	type Usage;

	unsafe fn from_res(pass: u32, res: &Resource, caches: &mut Caches, device: &Device) -> Self;

	unsafe fn add_read_usage(ty: &mut VirtualResourceData, pass: u32, usage: Self::Usage);
}

pub struct GpuData<'graph, T, U> {
	pub desc: T,
	pub write_usage: U,
	pub read_usages: ArenaMap<'graph, u32, U>,
}

pub enum GpuBufferType {
	Internal(usize),
	External(ExternalBuffer),
}

pub enum ImageType {
	Internal(ImageDesc),
	External(ExternalImage),
}

pub enum VirtualResourceType<'graph> {
	Data(NonNull<()>),
	UploadBuffer(GpuData<'graph, usize, BufferUsage>),
	GpuBuffer(GpuData<'graph, GpuBufferType, BufferUsage>),
	Image(GpuData<'graph, ImageType, ImageUsage>),
}

impl<'graph> VirtualResourceType<'graph> {
	unsafe fn upload_buffer(&mut self) -> &mut GpuData<'graph, usize, BufferUsage> {
		match self {
			VirtualResourceType::UploadBuffer(data) => data,
			_ => unreachable_unchecked(),
		}
	}

	unsafe fn gpu_buffer(&mut self) -> &mut GpuData<'graph, GpuBufferType, BufferUsage> {
		match self {
			VirtualResourceType::GpuBuffer(data) => data,
			_ => unreachable_unchecked(),
		}
	}

	unsafe fn image(&mut self) -> &mut GpuData<'graph, ImageType, ImageUsage> {
		match self {
			VirtualResourceType::Image(data) => data,
			_ => unreachable_unchecked(),
		}
	}
}

impl VirtualResource for UploadBufferHandle {
	type Usage = BufferUsage;

	unsafe fn from_res(_: u32, res: &Resource, _: &mut Caches, _: &Device) -> Self { res.upload_buffer() }

	unsafe fn add_read_usage(res: &mut VirtualResourceData, pass: u32, usage: Self::Usage) {
		res.ty.upload_buffer().read_usages.insert(pass, usage);
	}
}

impl VirtualResourceDesc for UploadBufferDesc {
	type Resource = UploadBufferHandle;

	fn ty(self, write_usage: BufferUsage, arena: &Arena) -> VirtualResourceType {
		VirtualResourceType::UploadBuffer(GpuData {
			desc: self.size,
			write_usage,
			read_usages: ArenaMap::with_hasher_in(BuildHasherDefault::default(), arena),
		})
	}
}

impl VirtualResource for GpuBufferHandle {
	type Usage = BufferUsage;

	unsafe fn from_res(_: u32, res: &Resource, _: &mut Caches, _: &Device) -> Self { res.gpu_buffer().resource.handle }

	unsafe fn add_read_usage(res: &mut VirtualResourceData, pass: u32, usage: Self::Usage) {
		res.ty.gpu_buffer().read_usages.insert(pass, usage);
	}
}

impl VirtualResourceDesc for GpuBufferDesc {
	type Resource = GpuBufferHandle;

	fn ty(self, write_usage: BufferUsage, arena: &Arena) -> VirtualResourceType {
		VirtualResourceType::GpuBuffer(GpuData {
			desc: GpuBufferType::Internal(self.size),
			write_usage,
			read_usages: ArenaMap::with_hasher_in(BuildHasherDefault::default(), arena),
		})
	}
}

impl VirtualResource for ImageView {
	type Usage = ImageUsage;

	unsafe fn from_res(pass: u32, res: &Resource, caches: &mut Caches, device: &Device) -> Self {
		let res = &res.image().resource;
		let usage = res.usages[&pass].access;

		caches
			.image_views
			.get(
				device,
				ImageViewDesc {
					image: res.handle,
					view_type: usage.view_type,
					format: usage.format,
					usage: match usage.usage {
						ImageUsageType::Storage(_) => ImageViewUsage::Storage,
						ImageUsageType::Sampled(_) => ImageViewUsage::Sampled,
						_ => ImageViewUsage::None,
					},
				},
			)
			.expect("Failed to create image view")
	}

	unsafe fn add_read_usage(res: &mut VirtualResourceData, pass: u32, usage: Self::Usage) {
		let image = res.ty.image();
		debug_assert!(
			compatible_formats(image.write_usage.format, usage.format),
			"`{:?}` and `{:?}` are not compatible",
			image.write_usage.format,
			usage.format
		);
		image.read_usages.insert(pass, usage);
	}
}

impl VirtualResourceDesc for ImageDesc {
	type Resource = ImageView;

	fn ty(self, usage: ImageUsage, arena: &Arena) -> VirtualResourceType {
		VirtualResourceType::Image(GpuData {
			desc: ImageType::Internal(self),
			write_usage: usage,
			read_usages: ArenaMap::with_hasher_in(BuildHasherDefault::default(), arena),
		})
	}
}

impl VirtualResourceDesc for ExternalBuffer {
	type Resource = GpuBufferHandle;

	fn ty(self, write_usage: BufferUsage, arena: &Arena) -> VirtualResourceType {
		VirtualResourceType::GpuBuffer(GpuData {
			desc: GpuBufferType::External(self),
			write_usage,
			read_usages: ArenaMap::with_hasher_in(Default::default(), arena),
		})
	}
}

impl VirtualResourceDesc for ExternalImage {
	type Resource = ImageView;

	fn ty(self, write_usage: ImageUsage, arena: &Arena) -> VirtualResourceType {
		VirtualResourceType::Image(GpuData {
			desc: ImageType::External(self),
			write_usage,
			read_usages: ArenaMap::with_hasher_in(Default::default(), arena),
		})
	}
}

pub fn compatible_formats(a: Format, b: Format) -> bool { get_format_block(a) == get_format_block(b) }

fn get_format_block(f: Format) -> i32 {
	macro_rules! f {
		($raw:ident,($i:ident)) => {
			Format::$i.as_raw() == $raw
		};
		($raw:ident,($f:ident : $t:ident)) => {
			(Format::$f.as_raw()..=Format::$t.as_raw()).contains(&$raw)
		};
	}

	macro_rules! select {
		($raw:ident, $($rest:tt)*) => {
			select!(# $raw, 0, $($rest)*)
		};

	    (# $raw:ident, $v:expr, $($tt:tt)||+, $($rest:tt)*) => {
			if $(f!($raw, $tt))||* {
				$v
			} else {
				select!(# $raw, $v + 1, $($rest)*)
			}
		};

		(# $raw:ident, $v:expr,) => {
			{
				$raw
			}
		};
	}

	let raw = f.as_raw();

	select! {
		raw,
		(R4G4_UNORM_PACK8) || (R8_UNORM:R8_SRGB),
		(R4G4B4A4_UNORM_PACK16:A1R5G5B5_UNORM_PACK16) || (R8G8_UNORM:R8G8_SRGB) || (R16_UNORM:R16_SFLOAT),
		(R8G8B8_UNORM:B8G8R8_SRGB),
		(R10X6G10X6_UNORM_2PACK16) || (R12X4G12X4_UNORM_2PACK16) || (R8G8B8A8_UNORM:A2B10G10R10_SINT_PACK32)
		|| (R16G16_UNORM:R16G16_SFLOAT) || (R32_UINT:R32_SFLOAT) || (B10G11R11_UFLOAT_PACK32:E5B9G9R9_UFLOAT_PACK32),
		(R16G16B16_UNORM:R16G16B16_SFLOAT),
		(R16G16B16A16_UNORM:R16G16B16A16_SFLOAT) || (R32G32_UINT:R32G32_SFLOAT) || (R64_UINT:R64_SFLOAT),
		(R32G32B32_UINT:R32G32B32_SFLOAT),
		(R32G32B32A32_UINT:R32G32B32A32_SFLOAT) || (R64G64_UINT:R64G64_SFLOAT),
		(R64G64B64_UINT:R64G64B64_SFLOAT),
		(R64G64B64A64_UINT:R64G64B64A64_SFLOAT),
		(BC1_RGB_UNORM_BLOCK:BC1_RGB_SRGB_BLOCK),
		(BC1_RGBA_UNORM_BLOCK:BC1_RGBA_SRGB_BLOCK),
		(BC2_UNORM_BLOCK:BC2_SRGB_BLOCK),
		(BC3_UNORM_BLOCK:BC3_SRGB_BLOCK),
		(BC4_UNORM_BLOCK:BC4_SNORM_BLOCK),
		(BC5_UNORM_BLOCK:BC5_SNORM_BLOCK),
		(BC6H_UFLOAT_BLOCK:BC6H_SFLOAT_BLOCK),
		(BC7_UNORM_BLOCK:BC7_SRGB_BLOCK),
		(ETC2_R8G8B8_UNORM_BLOCK:ETC2_R8G8B8_SRGB_BLOCK),
		(ETC2_R8G8B8A1_UNORM_BLOCK:ETC2_R8G8B8A1_SRGB_BLOCK),
		(ETC2_R8G8B8A8_UNORM_BLOCK:ETC2_R8G8B8A8_SRGB_BLOCK),
		(EAC_R11_UNORM_BLOCK:EAC_R11_SNORM_BLOCK),
		(EAC_R11G11_UNORM_BLOCK:EAC_R11G11_SNORM_BLOCK),
		(ASTC_4X4_UNORM_BLOCK:ASTC_4X4_SRGB_BLOCK) || (ASTC_4X4_SFLOAT_BLOCK),
		(ASTC_5X4_UNORM_BLOCK:ASTC_5X4_SRGB_BLOCK) || (ASTC_5X4_SFLOAT_BLOCK),
		(ASTC_5X5_UNORM_BLOCK:ASTC_5X5_SRGB_BLOCK) || (ASTC_5X5_SFLOAT_BLOCK),
		(ASTC_6X5_UNORM_BLOCK:ASTC_6X5_SRGB_BLOCK) || (ASTC_6X5_SFLOAT_BLOCK),
		(ASTC_6X6_UNORM_BLOCK:ASTC_6X6_SRGB_BLOCK) || (ASTC_6X6_SFLOAT_BLOCK),
		(ASTC_8X5_UNORM_BLOCK:ASTC_8X5_SRGB_BLOCK) || (ASTC_8X5_SFLOAT_BLOCK),
		(ASTC_8X6_UNORM_BLOCK:ASTC_8X6_SRGB_BLOCK) || (ASTC_8X6_SFLOAT_BLOCK),
		(ASTC_8X8_UNORM_BLOCK:ASTC_8X8_SRGB_BLOCK) || (ASTC_8X8_SFLOAT_BLOCK),
		(ASTC_10X5_UNORM_BLOCK:ASTC_10X5_SRGB_BLOCK) || (ASTC_10X5_SFLOAT_BLOCK),
		(ASTC_10X6_UNORM_BLOCK:ASTC_10X6_SRGB_BLOCK) || (ASTC_10X6_SFLOAT_BLOCK),
		(ASTC_10X8_UNORM_BLOCK:ASTC_10X8_SRGB_BLOCK) || (ASTC_10X8_SFLOAT_BLOCK),
		(ASTC_10X10_UNORM_BLOCK:ASTC_10X10_SRGB_BLOCK) || (ASTC_10X10_SFLOAT_BLOCK),
		(ASTC_12X10_UNORM_BLOCK:ASTC_12X10_SRGB_BLOCK) || (ASTC_12X10_SFLOAT_BLOCK),
		(ASTC_12X12_UNORM_BLOCK:ASTC_12X12_SRGB_BLOCK) || (ASTC_12X12_SFLOAT_BLOCK),
	}
}
