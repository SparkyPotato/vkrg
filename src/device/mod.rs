use std::sync::Mutex;

use ash::{
	extensions::{ext::DebugUtils, khr::Surface},
	vk::{DebugUtilsMessengerEXT, Fence, PhysicalDevice, Queue, SubmitInfo2},
	Entry,
	Instance,
};

use crate::Result;

mod init;

/// Has everything you need to do Vulkan stuff.
pub struct Device {
	entry: Entry,
	instance: Instance,
	debug_messenger: DebugUtilsMessengerEXT, // Can be null.
	physical_device: PhysicalDevice,
	device: ash::Device,
	surface_ext: Option<Surface>,
	debug_utils_ext: Option<DebugUtils>,
	queues: Queues<QueueData>,
}

struct QueueData {
	queue: Mutex<Queue>,
	family: u32,
}

enum Queues<T> {
	Separate {
		graphics: T, // Also supports presentation.
		compute: T,
		transfer: T,
	},
	Single(T),
}

impl<T> Queues<T> {
	fn map<U>(self, mut f: impl FnMut(T) -> U) -> Queues<U> {
		match self {
			Queues::Separate {
				graphics,
				compute,
				transfer,
			} => Queues::Separate {
				graphics: f(graphics),
				compute: f(compute),
				transfer: f(transfer),
			},
			Queues::Single(queue) => Queues::Single(f(queue)),
		}
	}
}

impl Device {
	pub fn entry(&self) -> &Entry { &self.entry }

	pub fn instance(&self) -> &Instance { &self.instance }

	pub fn device(&self) -> &ash::Device { &self.device }

	pub fn physical_device(&self) -> PhysicalDevice { self.physical_device }

	pub fn surface_ext(&self) -> Option<&Surface> { self.surface_ext.as_ref() }

	/// # SAFETY
	/// Thread-safety is handled, nothing else is.
	pub unsafe fn submit_graphics(&self, submits: &[SubmitInfo2], fence: Fence) -> Result<()> {
		match &self.queues {
			Queues::Single(graphics) => unsafe {
				self.device
					.queue_submit2(*graphics.queue.lock().unwrap(), submits, fence)
			}?,
			Queues::Separate { graphics, .. } => unsafe {
				self.device
					.queue_submit2(*graphics.queue.lock().unwrap(), submits, fence)
			}?,
		}

		Ok(())
	}

	/// # SAFETY
	/// Thread-safety is handled, nothing else is.
	pub unsafe fn submit_compute(&self, submits: &[SubmitInfo2], fence: Fence) -> Result<()> {
		match &self.queues {
			Queues::Single(compute) => unsafe {
				self.device
					.queue_submit2(*compute.queue.lock().unwrap(), submits, fence)
			}?,
			Queues::Separate { compute, .. } => unsafe {
				self.device
					.queue_submit2(*compute.queue.lock().unwrap(), submits, fence)
			}?,
		}

		Ok(())
	}

	/// # SAFETY
	/// Thread-safety is handled, nothing else is.
	pub unsafe fn submit_transfer(&self, submits: &[SubmitInfo2], fence: Fence) -> Result<()> {
		match &self.queues {
			Queues::Single(transfer) => unsafe {
				self.device
					.queue_submit2(*transfer.queue.lock().unwrap(), submits, fence)
			}?,
			Queues::Separate { transfer, .. } => unsafe {
				self.device
					.queue_submit2(*transfer.queue.lock().unwrap(), submits, fence)
			}?,
		}

		Ok(())
	}
}

impl Drop for Device {
	fn drop(&mut self) {
		unsafe {
			self.device.destroy_device(None);

			if let Some(utils) = self.debug_utils_ext.as_ref() {
				utils.destroy_debug_utils_messenger(self.debug_messenger, None);
			}
			self.instance.destroy_instance(None);
		}
	}
}
