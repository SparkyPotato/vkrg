use std::{
	alloc::{handle_alloc_error, AllocError, Allocator, Global, Layout},
	cell::UnsafeCell,
	ptr::{addr_of_mut, NonNull},
	slice::from_raw_parts_mut,
};

use tracing::trace;

pub fn collect_allocated_vec<T, A: Allocator>(iter: impl Iterator<Item = T>, alloc: A) -> Vec<T, A> {
	let mut vec = Vec::new_in(alloc);
	vec.extend(iter);
	vec
}

#[repr(C, align(8))]
struct Block {
	header: BlockHeader,
	data: [u8],
}

#[repr(align(8))]
struct BlockHeader {
	next: Option<NonNull<Block>>,
	offset: usize,
}

pub struct Arena {
	inner: UnsafeCell<Inner>,
}

struct Inner {
	head: NonNull<Block>,
	curr_block: NonNull<Block>,
	alloc_count: usize,
	last_alloc: NonNull<u8>,
}

impl Arena {
	pub fn new() -> Self { Self::with_block_size(1024 * 1024) }

	pub fn with_block_size(block_size: usize) -> Self {
		let head = match Self::allocate_block(block_size) {
			Ok(head) => head,
			Err(_) => handle_alloc_error(Self::block_layout(block_size)),
		};

		Arena {
			inner: UnsafeCell::new(Inner {
				head,
				curr_block: head,
				alloc_count: 0,
				last_alloc: NonNull::dangling(),
			}),
		}
	}

	pub fn reset(&mut self) {
		assert_eq!(
			self.inner.get_mut().alloc_count,
			0,
			"tried to reset Arena with living allocations"
		);
	}

	unsafe fn reset_all_blocks(&self) {
		let inner = self.inner.get();
		let mut block = Some((*inner).head);
		while let Some(mut b) = block {
			let mut prev = b;
			block = b.as_mut().header.next;
			prev.as_mut().header.offset = 0;
		}
	}

	fn block_layout(size: usize) -> Layout {
		unsafe {
			Layout::from_size_align_unchecked(
				std::mem::size_of::<BlockHeader>() + size,
				std::mem::align_of::<BlockHeader>(),
			)
		}
	}

	fn allocate_block(size: usize) -> Result<NonNull<Block>, AllocError> {
		unsafe {
			let mut head: NonNull<Block> = Global
				.allocate(Self::block_layout(size))
				.map(|ptr| NonNull::new_unchecked(std::ptr::from_raw_parts_mut(ptr.as_ptr() as *mut (), size)))?;

			addr_of_mut!(head.as_mut().header).write(BlockHeader { next: None, offset: 0 });

			Ok(head)
		}
	}

	fn extend(&self, size: usize) -> Result<NonNull<Block>, AllocError> {
		trace!("arena: block is full, allocating new block");

		let inner = self.inner.get();
		let new = Self::allocate_block(size)?;
		unsafe {
			(*inner).curr_block.as_mut().header.next = Some(new);
			(*inner).curr_block = new;
		}

		Ok(new)
	}

	fn aligned_offset(&self, align: usize) -> usize {
		unsafe {
			let curr = (*self.inner.get()).curr_block.as_ref();
			let base = curr.data.as_ptr();
			let unaligned = base.add(curr.header.offset) as usize;
			let aligned = (unaligned + align - 1) & !(align - 1);
			aligned - base as usize
		}
	}
}

unsafe impl Allocator for Arena {
	fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
		// SAFETY: I said so (and miri agrees).
		unsafe {
			let inner = self.inner.get();

			let ret = if layout.size() > (*inner).curr_block.as_ref().data.len() {
				// Allocate a dedicated block for this, since it's too big for our current block size.
				Ok(self.extend(layout.size())?.as_mut().data.into())
			} else {
				let offset = self.aligned_offset(layout.align());

				let target = if offset + layout.size() > (*inner).curr_block.as_ref().data.len() {
					// There's not enough space in the current block, so go to the next one.
					if let Some(next) = (*inner).curr_block.as_mut().header.next {
						// There's a next block, so we can use it.
						(*inner).curr_block = next;
					} else {
						// There's no next block, so we need to allocate a new one.
						self.extend((*inner).curr_block.as_ref().data.len())?;
					}

					let offset = self.aligned_offset(layout.align());
					(*inner).curr_block.as_mut().data.as_mut_ptr().add(offset)
				} else {
					// There's enough space in the current block, so use it.
					(*inner).curr_block.as_mut().data.as_mut_ptr().add(offset)
				};

				(*inner).curr_block.as_mut().header.offset += layout.size();
				Ok(NonNull::new_unchecked(from_raw_parts_mut(target, layout.size())))
			};

			match ret {
				Ok(ptr) => {
					(*inner).alloc_count += 1;
					(*inner).last_alloc = ptr.cast();
					Ok(ptr)
				},
				x => x,
			}
		}
	}

	unsafe fn deallocate(&self, ptr: NonNull<u8>, _: Layout) {
		let inner = self.inner.get();

		(*inner).alloc_count -= 1;
		if (*inner).alloc_count == 0 {
			self.reset_all_blocks()
		} else if ptr == (*inner).last_alloc {
			let offset = ptr.as_ptr().offset_from((*inner).curr_block.as_ref().data.as_ptr());
			(*inner).curr_block.as_mut().header.offset = offset as _;
		}
	}

	unsafe fn grow(
		&self, ptr: NonNull<u8>, old_layout: Layout, new_layout: Layout,
	) -> Result<NonNull<[u8]>, AllocError> {
		let inner = self.inner.get();

		if ptr == (*inner).last_alloc {
			let offset = ptr.as_ptr().offset_from((*inner).curr_block.as_ref().data.as_ptr());
			let new_offset = offset as usize + new_layout.size();
			if new_offset <= (*inner).curr_block.as_ref().data.len() {
				(*inner).curr_block.as_mut().header.offset = new_offset;
				Ok(NonNull::new_unchecked(from_raw_parts_mut(
					ptr.as_ptr(),
					new_layout.size(),
				)))
			} else {
				let new_ptr = self.allocate(new_layout)?;

				std::ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_ptr() as *mut _, old_layout.size());
				self.deallocate(ptr, old_layout);

				Ok(new_ptr)
			}
		} else {
			let new_ptr = self.allocate(new_layout)?;

			std::ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_ptr() as *mut _, old_layout.size());
			self.deallocate(ptr, old_layout);

			Ok(new_ptr)
		}
	}
}

impl Drop for Arena {
	fn drop(&mut self) {
		let inner = self.inner.get_mut();

		let mut block = Some(inner.head);
		while let Some(mut b) = block {
			let mut prev = b;
			unsafe {
				block = b.as_mut().header.next;
				Global.deallocate(prev.cast(), Self::block_layout(prev.as_mut().data.len()));
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn non_overlapping() {
		let arena = Arena::new();

		unsafe {
			let a = arena.allocate(Layout::new::<u32>()).unwrap().as_ptr() as *mut u32;
			let b = arena.allocate(Layout::new::<u32>()).unwrap().as_ptr() as *mut u32;

			*a = 0xDEADBEEF;
			*b = 0xCAFEBABE;

			assert_eq!(*a, 0xDEADBEEF);
			assert_eq!(*b, 0xCAFEBABE);
		}
	}

	#[test]
	fn allocate_over_size() {
		let arena = Arena::with_block_size(256);

		let vec = Vec::<u8, &Arena>::with_capacity_in(178, &arena);
		let vec = Vec::<u8, &Arena>::with_capacity_in(128, &arena);
	}

	#[test]
	fn reset() {
		let arena = Arena::new();

		unsafe {
			let a = arena.allocate(Layout::new::<u32>()).unwrap().as_ptr() as *mut u32;
			arena.reset_all_blocks();
			let b = arena.allocate(Layout::new::<u32>()).unwrap().as_ptr() as *mut u32;

			*a = 0xDEADBEEF;
			*b = 0xCAFEBABE;

			assert_eq!(*a, 0xCAFEBABE);
			assert_eq!(*b, 0xCAFEBABE);
		}
	}

	#[test]
	#[should_panic]
	fn early_reset() {
		let mut arena = Arena::new();

		let _ = arena.allocate(Layout::new::<u32>()).unwrap().as_ptr() as *mut u32;
		arena.reset();
	}

	#[test]
	fn grow() {
		let arena = Arena::new();

		unsafe {
			let a = arena.allocate(Layout::new::<u32>()).unwrap().cast();
			let b = arena
				.grow(a, Layout::new::<u32>(), Layout::new::<[u32; 2]>())
				.unwrap()
				.cast();
			assert_eq!(a, b);
		}
	}
}
