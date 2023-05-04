use ash::vk::{
	CommandBuffer,
	CommandBufferAllocateInfo,
	CommandBufferBeginInfo,
	CommandBufferLevel,
	CommandBufferSubmitInfo,
	CommandBufferUsageFlags,
	CommandPool,
	CommandPoolCreateFlags,
	CommandPoolCreateInfo,
	CommandPoolResetFlags,
	Fence,
	PipelineStageFlags2,
	Semaphore,
	SemaphoreCreateInfo,
	SemaphoreSubmitInfo,
	SemaphoreType,
	SemaphoreTypeCreateInfo,
	SemaphoreWaitInfo,
	SubmitInfo2,
};

use crate::{
	arena::{Arena, IteratorAlloc},
	device::Device,
	graph::{Access, ExternalSync},
	Result,
};

/// A timeline semaphore that keeps track of the current value.
pub struct TimelineSemaphore {
	inner: Semaphore,
	value: u64,
}

impl TimelineSemaphore {
	pub fn new(device: &Device) -> Result<Self> {
		let semaphore = unsafe {
			device.device().create_semaphore(
				&SemaphoreCreateInfo::builder().push_next(
					&mut SemaphoreTypeCreateInfo::builder()
						.semaphore_type(SemaphoreType::TIMELINE)
						.initial_value(0),
				),
				None,
			)
		}?;

		Ok(Self {
			inner: semaphore,
			value: 0,
		})
	}

	pub fn wait(&self, device: &Device) -> Result<()> {
		unsafe {
			device
				.device()
				.wait_semaphores(
					&SemaphoreWaitInfo::builder()
						.semaphores(&[self.inner])
						.values(&[self.value]),
					u64::MAX,
				)
				.map_err(Into::into)
		}
	}

	pub fn next(&mut self) -> (Semaphore, u64) {
		self.value += 1;
		(self.inner, self.value)
	}

	pub fn value(&self) -> u64 { self.value }

	pub fn semaphore(&self) -> Semaphore { self.inner }

	pub fn destroy(self, device: &Device) {
		unsafe {
			self.wait(device).expect("Failed to wait for semaphore");
			device.device().destroy_semaphore(self.inner, None);
		}
	}
}

pub struct FrameData {
	pub semaphore: TimelineSemaphore,
	pool: CommandPool,
	bufs: Vec<CommandBuffer>,
	buf_cursor: usize,
}

impl FrameData {
	pub fn new(device: &Device) -> Result<Self> {
		let pool = unsafe {
			device.device().create_command_pool(
				&CommandPoolCreateInfo::builder()
					.queue_family_index(*device.queue_families().graphics())
					.flags(CommandPoolCreateFlags::TRANSIENT),
				None,
			)
		}?;

		Ok(Self {
			pool,
			semaphore: TimelineSemaphore::new(device)?,
			bufs: Vec::new(),
			buf_cursor: 0,
		})
	}

	pub fn reset(&mut self, device: &Device) -> Result<()> {
		unsafe {
			// Let GPU finish this frame before doing anything else.
			self.semaphore.wait(device)?;

			device
				.device()
				.reset_command_pool(self.pool, CommandPoolResetFlags::empty())?;
			self.buf_cursor = 0; // We can now hand out the first buffer again.

			Ok(())
		}
	}

	pub fn cmd_buf(&mut self, device: &Device) -> Result<CommandBuffer> {
		if let Some(buf) = self.bufs.get(self.buf_cursor) {
			self.buf_cursor += 1;
			Ok(*buf)
		} else {
			let buf = unsafe {
				device.device().allocate_command_buffers(
					&CommandBufferAllocateInfo::builder()
						.command_pool(self.pool)
						.level(CommandBufferLevel::PRIMARY)
						.command_buffer_count(1),
				)
			}?[0];

			self.bufs.push(buf);

			Ok(buf)
		}
	}

	pub fn destroy(self, device: &Device) {
		unsafe {
			self.semaphore.destroy(device);
			device.device().destroy_command_pool(self.pool, None);
		}
	}
}

pub struct Submitter<'a> {
	data: &'a mut FrameData,
	buf: CommandBuffer,
	cached_wait: Vec<SemaphoreSubmitInfo, &'a Arena>,
	cached_signal: Vec<SemaphoreSubmitInfo, &'a Arena>,
}

impl<'a> Submitter<'a> {
	pub fn new(arena: &'a Arena, frames: &'a mut [FrameData], curr_frame: usize) -> Self {
		let wait = &frames[curr_frame ^ 1].semaphore;

		Self {
			cached_wait: std::iter::once(
				SemaphoreSubmitInfo::builder()
					.stage_mask(PipelineStageFlags2::TOP_OF_PIPE)
					.semaphore(wait.inner)
					.value(wait.value)
					.build(),
			)
			.collect_in(arena),
			data: &mut frames[curr_frame],
			buf: CommandBuffer::null(),
			cached_signal: Vec::new_in(arena),
		}
	}

	pub fn pass(
		&mut self, device: &Device, pre_sync: &[ExternalSync<Access>], post_sync: &[ExternalSync<Access>],
	) -> Result<CommandBuffer> {
		match (!pre_sync.is_empty(), !post_sync.is_empty()) {
			(false, false) => self.ensure_buf(device)?,
			(true, false) => {
				self.submit(&device)?;
				self.ensure_buf(device)?;
				self.cached_wait.extend(sync_to_info(pre_sync));
			},
			(false, true) => {
				if !self.cached_signal.is_empty() {
					self.submit(&device)?;
					self.ensure_buf(device)?;
				}
				self.cached_signal.extend(sync_to_info(post_sync));
			},
			(true, true) => {
				self.submit(&device)?;
				self.ensure_buf(device)?;
				self.cached_wait.extend(sync_to_info(pre_sync));
				self.cached_signal.extend(sync_to_info(post_sync));
			},
		}

		Ok(self.buf)
	}

	pub fn finish(mut self, device: &Device) -> Result<()> {
		let (sem, set) = self.data.semaphore.next();
		self.cached_signal.push(
			SemaphoreSubmitInfo::builder()
				.stage_mask(PipelineStageFlags2::BOTTOM_OF_PIPE)
				.semaphore(sem)
				.value(set)
				.build(),
		);
		self.submit(device)
	}

	fn submit(&mut self, device: &Device) -> Result<()> {
		unsafe {
			if self.buf != CommandBuffer::null() {
				device.device().end_command_buffer(self.buf)?;
				let ret = device.submit_graphics(
					&[SubmitInfo2::builder()
						.wait_semaphore_infos(&self.cached_wait)
						.command_buffer_infos(&[CommandBufferSubmitInfo::builder().command_buffer(self.buf).build()])
						.signal_semaphore_infos(&self.cached_signal)
						.build()],
					Fence::null(),
				);

				self.cached_wait.clear();
				self.cached_signal.clear();
				self.buf = CommandBuffer::null();

				ret
			} else {
				Ok(())
			}
		}
	}

	fn ensure_buf(&mut self, device: &Device) -> Result<()> {
		if self.buf == CommandBuffer::null() {
			self.buf = self.data.cmd_buf(device)?;
			unsafe {
				device.device().begin_command_buffer(
					self.buf,
					&CommandBufferBeginInfo::builder().flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT),
				)?;
			}
		}

		Ok(())
	}
}

fn sync_to_info<'a>(
	sync: impl IntoIterator<Item = &'a ExternalSync<Access>> + 'a,
) -> impl Iterator<Item = SemaphoreSubmitInfo> + 'a {
	sync.into_iter().map(|sync| {
		SemaphoreSubmitInfo::builder()
			.stage_mask(sync.access.stage)
			.semaphore(sync.semaphore)
			.value(sync.value)
			.build()
	})
}
