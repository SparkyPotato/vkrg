use helpers::{compile, run, simple_pipeline, App, ShaderStage};
use vkrg::{
	ash::vk::{
		AttachmentLoadOp,
		AttachmentStoreOp,
		ClearColorValue,
		ClearValue,
		Extent2D,
		Format,
		ImageLayout,
		ImageViewType,
		Pipeline,
		PipelineBindPoint,
		Rect2D,
		RenderingAttachmentInfo,
		RenderingInfo,
		Viewport,
	},
	device::Device,
	graph::{ExternalImage, Frame, ImageUsage, ImageUsageType, ReadId},
	resource::ImageView,
};

struct Triangle(Pipeline);

impl App for Triangle {
	const NAME: &'static str = "triangle";

	fn create(device: &Device) -> Self {
		let vertex = compile(include_str!("vertex.wgsl"), ShaderStage::Vertex);
		let fragment = compile(include_str!("fragment.wgsl"), ShaderStage::Fragment);
		let pipeline = simple_pipeline(device, &vertex, &fragment, Format::B8G8R8A8_SRGB);

		Self(pipeline)
	}

	fn destroy(self, device: &Device) {
		unsafe {
			device.device().destroy_pipeline(self.0, None);
		}
	}

	fn render<'frame>(
		&'frame mut self, frame: &mut Frame<'frame, '_>, width: u32, height: u32, window: ExternalImage,
	) -> ReadId<ImageView> {
		let mut pass = frame.pass("triangle");
		let (read, write) = pass.output(
			window,
			ImageUsage {
				format: Format::B8G8R8A8_SRGB,
				usage: ImageUsageType::ColorAttachment,
				view_type: ImageViewType::TYPE_2D,
			},
		);

		pass.build(move |mut ctx| unsafe {
			let view = ctx.write(write);
			ctx.device.device().cmd_begin_rendering(
				ctx.buf,
				&RenderingInfo::builder()
					.render_area(
						Rect2D::builder()
							.extent(Extent2D::builder().width(width).height(height).build())
							.build(),
					)
					.layer_count(1)
					.color_attachments(&[RenderingAttachmentInfo::builder()
						.image_view(view.view)
						.image_layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
						.load_op(AttachmentLoadOp::CLEAR)
						.clear_value(ClearValue {
							color: ClearColorValue {
								float32: [0.0, 0.0, 0.0, 1.0],
							},
						})
						.store_op(AttachmentStoreOp::STORE)
						.build()]),
			);

			ctx.device
				.device()
				.cmd_bind_pipeline(ctx.buf, PipelineBindPoint::GRAPHICS, self.0);
			ctx.device.device().cmd_set_viewport(
				ctx.buf,
				0,
				&[Viewport {
					x: 0.0,
					y: 0.0,
					width: width as f32,
					height: height as f32,
					min_depth: 0.0,
					max_depth: 1.0,
				}],
			);
			ctx.device.device().cmd_set_scissor(
				ctx.buf,
				0,
				&[Rect2D::builder()
					.extent(Extent2D::builder().width(width).height(height).build())
					.build()],
			);

			ctx.device.device().cmd_draw(ctx.buf, 3, 1, 0, 0);

			ctx.device.device().cmd_end_rendering(ctx.buf);
		});

		read
	}
}

fn main() { run::<Triangle>() }
