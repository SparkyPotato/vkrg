use ash::vk::{
	CommandBuffer,
	CommandBufferAllocateInfo,
	CommandBufferLevel,
	CommandPool,
	CommandPoolCreateFlags,
	CommandPoolCreateInfo,
	CommandPoolResetFlags,
	Semaphore,
	SemaphoreCreateInfo,
	SemaphoreType,
	SemaphoreTypeCreateInfo,
	SemaphoreWaitInfo,
};

use crate::{device::Device, Result};

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
		let val = self.value;
		self.value += 1;
		(self.inner, val)
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
