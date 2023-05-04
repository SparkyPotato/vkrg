use std::{ffi::CStr, mem::ManuallyDrop};

pub use naga::ShaderStage;
use naga::{
	back::{spv, spv::PipelineOptions},
	front::wgsl,
	valid,
	valid::{Capabilities, ValidationFlags},
};
use vkrg::{
	arena::Arena,
	ash::{
		extensions::khr,
		vk::{
			AccessFlags2,
			BlendFactor,
			BlendOp,
			ColorComponentFlags,
			ColorSpaceKHR,
			CompositeAlphaFlagsKHR,
			CullModeFlags,
			DynamicState,
			Extent2D,
			Fence,
			Format,
			FrontFace,
			GraphicsPipelineCreateInfo,
			Image,
			ImageLayout,
			ImageUsageFlags,
			Pipeline,
			PipelineCache,
			PipelineColorBlendAttachmentState,
			PipelineColorBlendStateCreateInfo,
			PipelineDynamicStateCreateInfo,
			PipelineInputAssemblyStateCreateInfo,
			PipelineLayoutCreateInfo,
			PipelineMultisampleStateCreateInfo,
			PipelineRasterizationStateCreateInfo,
			PipelineRenderingCreateInfo,
			PipelineShaderStageCreateInfo,
			PipelineStageFlags2,
			PipelineVertexInputStateCreateInfo,
			PipelineViewportStateCreateInfo,
			PolygonMode,
			PresentInfoKHR,
			PresentModeKHR,
			PrimitiveTopology,
			Rect2D,
			SampleCountFlags,
			Semaphore,
			SemaphoreCreateInfo,
			ShaderModuleCreateInfo,
			ShaderStageFlags,
			SharingMode,
			SurfaceKHR,
			SurfaceTransformFlagsKHR,
			SwapchainCreateInfoKHR,
			SwapchainKHR,
			Viewport,
		},
	},
	device::Device,
	graph::{Access, ExternalImage, ExternalSync, Frame, ImageAccess, ReadId, RenderGraph},
	resource::ImageView,
};
use winit::{
	event::{Event, WindowEvent},
	event_loop::{ControlFlow, EventLoop},
	window::{Window, WindowBuilder, WindowButtons},
};

pub trait App: 'static + Sized {
	const NAME: &'static str;

	fn create(device: &Device) -> Self;

	fn destroy(self, device: &Device);

	fn render<'frame>(
		&'frame mut self, frame: &mut Frame<'frame, '_>, width: u32, height: u32, window: ExternalImage,
	) -> ReadId<ImageView>;
}

pub fn run<T: App>() -> ! {
	let event_loop = EventLoop::new();
	let window = WindowBuilder::new()
		.with_title(format!("Example: {}", T::NAME))
		.with_resizable(false)
		.with_enabled_buttons(WindowButtons::CLOSE | WindowButtons::MINIMIZE)
		.build(&event_loop)
		.unwrap();

	let (device, surface) = unsafe {
		Device::with_window_and_layers_and_extensions(&window, &event_loop, &[], &[], &[khr::Swapchain::name()])
			.unwrap()
	};
	let swapchain = Swapchain::new(&device, surface, &window);

	let mut arena = Arena::new();
	let mut graph = ManuallyDrop::new(RenderGraph::new(&device).unwrap());

	let mut app = ManuallyDrop::new(T::create(&device));

	event_loop.run(move |event, _, flow| match event {
		Event::MainEventsCleared => window.request_redraw(),
		Event::RedrawRequested(_) => {
			arena.reset();
			let mut frame = graph.frame(&arena);

			let (image, id) = swapchain.acquire();

			let size = window.inner_size();
			let _ = app.render(&mut frame, size.width, size.height, image);
			frame.run(&device).unwrap();

			swapchain.present(&device, id);
		},
		Event::WindowEvent { event, .. } => match event {
			WindowEvent::CloseRequested => *flow = ControlFlow::Exit,
			_ => {},
		},
		Event::LoopDestroyed => unsafe {
			ManuallyDrop::take(&mut graph).destroy(&device);
			ManuallyDrop::take(&mut app).destroy(&device);
			device.surface_ext().unwrap().destroy_surface(surface, None);
		},
		_ => {},
	})
}

struct Swapchain {
	swapchain_ext: khr::Swapchain,
	swapchain: SwapchainKHR,
	images: Vec<Image>,
	available: Semaphore,
	rendered: Semaphore,
}

impl Swapchain {
	fn new(device: &Device, surface: SurfaceKHR, window: &Window) -> Self {
		unsafe {
			let swapchain_ext = khr::Swapchain::new(device.instance(), device.device());
			let swapchain = swapchain_ext
				.create_swapchain(
					&SwapchainCreateInfoKHR::builder()
						.surface(surface)
						.min_image_count(2)
						.image_format(Format::B8G8R8A8_SRGB)
						.image_color_space(ColorSpaceKHR::SRGB_NONLINEAR)
						.image_extent(Extent2D {
							width: window.inner_size().width,
							height: window.inner_size().height,
						})
						.image_array_layers(1)
						.image_usage(ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::TRANSFER_DST)
						.image_sharing_mode(SharingMode::EXCLUSIVE)
						.pre_transform(SurfaceTransformFlagsKHR::IDENTITY)
						.composite_alpha(CompositeAlphaFlagsKHR::OPAQUE)
						.present_mode(PresentModeKHR::FIFO)
						.clipped(true),
					None,
				)
				.unwrap();

			let images = swapchain_ext.get_swapchain_images(swapchain).unwrap();

			Self {
				swapchain_ext,
				swapchain,
				images,
				available: device
					.device()
					.create_semaphore(&SemaphoreCreateInfo::builder().build(), None)
					.unwrap(),
				rendered: device
					.device()
					.create_semaphore(&SemaphoreCreateInfo::builder().build(), None)
					.unwrap(),
			}
		}
	}

	fn acquire(&self) -> (ExternalImage, u32) {
		unsafe {
			let (id, _) = self
				.swapchain_ext
				.acquire_next_image(self.swapchain, u64::MAX, self.available, Fence::null())
				.unwrap();
			(
				ExternalImage {
					handle: self.images[id as usize],
					pre_sync: ExternalSync {
						semaphore: self.available,
						value: 0,
						access: ImageAccess {
							access: Access {
								stage: PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
								access: AccessFlags2::NONE,
							},
							layout: ImageLayout::PRESENT_SRC_KHR,
						},
					},
					post_sync: ExternalSync {
						semaphore: self.rendered,
						value: 0,
						access: Access::default(),
					},
				},
				id,
			)
		}
	}

	fn present(&self, device: &Device, id: u32) {
		unsafe {
			self.swapchain_ext
				.queue_present(
					*device.graphics_queue(),
					&PresentInfoKHR::builder()
						.wait_semaphores(&[self.rendered])
						.swapchains(&[self.swapchain])
						.image_indices(&[id])
						.build(),
				)
				.unwrap();
		}
	}
}

impl Drop for Swapchain {
	fn drop(&mut self) {
		unsafe {
			self.swapchain_ext.destroy_swapchain(self.swapchain, None);
		}
	}
}

// We use WGSL because there's a nice compiler in rust for it.
pub fn compile(shader: &str, stage: ShaderStage) -> Vec<u32> {
	let module = wgsl::parse_str(shader).map_err(|x| x.emit_to_stderr(shader)).unwrap();
	let info = valid::Validator::new(ValidationFlags::all(), Capabilities::all())
		.validate(&module)
		.unwrap();
	spv::write_vec(
		&module,
		&info,
		&Default::default(),
		Some(&PipelineOptions {
			entry_point: "main".into(),
			shader_stage: stage,
		}),
	)
	.unwrap()
}

pub fn simple_pipeline(device: &Device, vertex: &[u32], fragment: &[u32], format: Format) -> Pipeline {
	unsafe {
		let vertex = device
			.device()
			.create_shader_module(&ShaderModuleCreateInfo::builder().code(vertex), None)
			.unwrap();
		let fragment = device
			.device()
			.create_shader_module(&ShaderModuleCreateInfo::builder().code(fragment), None)
			.unwrap();

		let layout = device
			.device()
			.create_pipeline_layout(
				&PipelineLayoutCreateInfo::builder().set_layouts(&[device.base_descriptors().layout()]),
				None,
			)
			.unwrap();

		let ret = device
			.device()
			.create_graphics_pipelines(
				PipelineCache::null(),
				&[GraphicsPipelineCreateInfo::builder()
					.stages(&[
						PipelineShaderStageCreateInfo::builder()
							.stage(ShaderStageFlags::VERTEX)
							.module(vertex)
							.name(CStr::from_bytes_with_nul_unchecked(b"main\0"))
							.build(),
						PipelineShaderStageCreateInfo::builder()
							.stage(ShaderStageFlags::FRAGMENT)
							.module(fragment)
							.name(CStr::from_bytes_with_nul_unchecked(b"main\0"))
							.build(),
					])
					.vertex_input_state(&PipelineVertexInputStateCreateInfo::builder())
					.input_assembly_state(
						&PipelineInputAssemblyStateCreateInfo::builder().topology(PrimitiveTopology::TRIANGLE_LIST),
					)
					.viewport_state(
						&PipelineViewportStateCreateInfo::builder()
							.viewports(&[Viewport::builder().build()])
							.scissors(&[Rect2D::builder().build()]),
					)
					.rasterization_state(
						&PipelineRasterizationStateCreateInfo::builder()
							.polygon_mode(PolygonMode::FILL)
							.front_face(FrontFace::COUNTER_CLOCKWISE)
							.cull_mode(CullModeFlags::NONE),
					)
					.multisample_state(
						&PipelineMultisampleStateCreateInfo::builder().rasterization_samples(SampleCountFlags::TYPE_1),
					)
					.color_blend_state(
						&PipelineColorBlendStateCreateInfo::builder().attachments(&[
							PipelineColorBlendAttachmentState::builder()
								.color_write_mask(
									ColorComponentFlags::R
										| ColorComponentFlags::G | ColorComponentFlags::B
										| ColorComponentFlags::A,
								)
								.blend_enable(true)
								.src_color_blend_factor(BlendFactor::SRC_ALPHA)
								.dst_color_blend_factor(BlendFactor::ONE_MINUS_SRC_ALPHA)
								.color_blend_op(BlendOp::ADD)
								.src_alpha_blend_factor(BlendFactor::ONE)
								.dst_alpha_blend_factor(BlendFactor::ONE_MINUS_SRC_ALPHA)
								.alpha_blend_op(BlendOp::ADD)
								.build(),
						]),
					)
					.dynamic_state(
						&PipelineDynamicStateCreateInfo::builder()
							.dynamic_states(&[DynamicState::VIEWPORT, DynamicState::SCISSOR]),
					)
					.layout(layout)
					.push_next(&mut PipelineRenderingCreateInfo::builder().color_attachment_formats(&[format]))
					.build()],
				None,
			)
			.unwrap()[0];

		device.device().destroy_shader_module(vertex, None);
		device.device().destroy_shader_module(fragment, None);
		device.device().destroy_pipeline_layout(layout, None);

		ret
	}
}
