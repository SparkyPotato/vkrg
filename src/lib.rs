#![feature(allocator_api)]
#![feature(ptr_metadata)]

use std::fmt::{Debug, Display};

pub mod arena;
pub mod device;
pub mod graph;

#[derive(Clone)]
pub enum Error {
	Message(String),
	Vulkan(ash::vk::Result),
}

impl Display for Error {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		match self {
			Error::Message(msg) => write!(f, "{}", msg),
			Error::Vulkan(res) => write!(f, "Vulkan error: {}", res),
		}
	}
}

impl Debug for Error {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { <Self as Display>::fmt(self, f) }
}

impl From<String> for Error {
	fn from(message: String) -> Self { Error::Message(message) }
}

impl From<ash::vk::Result> for Error {
	fn from(result: ash::vk::Result) -> Self { Error::Vulkan(result) }
}

pub type Result<T> = std::result::Result<T, Error>;
