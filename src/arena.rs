use std::{
	alloc::{handle_alloc_error, AllocError, Allocator, Global, Layout},
	cell::UnsafeCell,
	ptr::{addr_of_mut, NonNull},
};

use tracing::trace;

pub fn collect_allocated_vec<T, A: Allocator>(iter: impl IntoIterator<Item = T>, alloc: A) -> Vec<T, A> {
	let iter = iter.into_iter();
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
	last_alloc: usize,
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
				last_alloc: 0,
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
		while let Some(b) = block {
			let prev = b;
			block = (*b.as_ptr()).header.next;
			(*prev.as_ptr()).header.offset = 0;
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
			let head: NonNull<Block> = Global
				.allocate(Self::block_layout(size))
				.map(|ptr| NonNull::new_unchecked(std::ptr::from_raw_parts_mut(ptr.as_ptr() as *mut (), size)))?;

			addr_of_mut!((*head.as_ptr()).header).write(BlockHeader { next: None, offset: 0 });

			Ok(head)
		}
	}

	fn extend(&self, size: usize) -> Result<NonNull<Block>, AllocError> {
		trace!("arena: block is full, allocating new block");

		let inner = self.inner.get();
		let new = Self::allocate_block(size)?;
		unsafe {
			(*(*inner).curr_block.as_ptr()).header.next = Some(new);
			(*inner).curr_block = new;
		}

		Ok(new)
	}

	fn aligned_offset(&self, align: usize) -> usize {
		unsafe {
			let curr = (*self.inner.get()).curr_block.as_ptr();
			let base = (*curr).data.as_ptr();
			let unaligned = base.add((*curr).header.offset) as usize;
			let aligned = (unaligned + align - 1) & !(align - 1);
			aligned - base as usize
		}
	}
}

unsafe impl Allocator for Arena {
	fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
		// SAFETY: Uh.
		unsafe {
			let inner = self.inner.get();

			let (ptr, offset) = if layout.size() > (*(*inner).curr_block.as_ptr()).data.len() {
				// Allocate a dedicated block for this, since it's too big for our current block size.
				let ptr = addr_of_mut!((*self.extend(layout.size())?.as_ptr()).data).cast();
				(ptr, layout.size())
			} else {
				let mut offset = self.aligned_offset(layout.align());
				if offset + layout.size() > (*(*inner).curr_block.as_ptr()).data.len() {
					// There's not enough space in the current block, so go to the next one.
					if let Some(next) = (*(*inner).curr_block.as_ptr()).header.next {
						// There's a next block, so we can use it.
						(*inner).curr_block = next;
					} else {
						// There's no next block, so we need to allocate a new one.
						self.extend((*(*inner).curr_block.as_ptr()).data.len())?;
					}

					offset = self.aligned_offset(layout.align());
				}

				let target = addr_of_mut!((*(*inner).curr_block.as_ptr()).data)
					.cast::<u8>()
					.add(offset);
				(target, offset + layout.size())
			};

			(*inner).alloc_count += 1;
			(*inner).last_alloc = ptr.to_raw_parts().0.addr();
			(*(*inner).curr_block.as_ptr()).header.offset = offset;

			Ok(NonNull::new_unchecked(std::ptr::from_raw_parts_mut(
				ptr as _,
				layout.size(),
			)))
		}
	}

	unsafe fn deallocate(&self, ptr: NonNull<u8>, _: Layout) {
		let inner = self.inner.get();

		(*inner).alloc_count -= 1;
		if (*inner).alloc_count == 0 {
			self.reset_all_blocks()
		} else if ptr.addr().get() == (*inner).last_alloc {
			let offset = ptr.as_ptr().offset_from((*(*inner).curr_block.as_ptr()).data.as_ptr());
			(*(*inner).curr_block.as_ptr()).header.offset = offset as _;
		}
	}

	unsafe fn grow(
		&self, ptr: NonNull<u8>, old_layout: Layout, new_layout: Layout,
	) -> Result<NonNull<[u8]>, AllocError> {
		let inner = self.inner.get();

		if ptr.addr().get() == (*inner).last_alloc {
			// Reuse the last allocation if possible.
			let offset = ptr.as_ptr().offset_from((*(*inner).curr_block.as_ptr()).data.as_ptr());
			let new_offset = offset as usize + new_layout.size();
			if new_offset <= (*(*inner).curr_block.as_ptr()).data.len() {
				(*(*inner).curr_block.as_ptr()).header.offset = new_offset;
				return Ok(NonNull::new_unchecked(std::ptr::from_raw_parts_mut(
					ptr.as_ptr() as _,
					new_layout.size(),
				)));
			}
		}

		let new_ptr = self.allocate(new_layout)?;
		std::ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_ptr() as *mut _, old_layout.size());
		(*inner).alloc_count -= 1;
		Ok(new_ptr)
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

			*a = 123;
			*b = 456;

			assert_eq!(*a, 123);
			assert_eq!(*b, 456);
		}
	}

	#[test]
	fn allocate_over_size() {
		let arena = Arena::with_block_size(256);

		let _vec = Vec::<u8, &Arena>::with_capacity_in(178, &arena);
		let _vec = Vec::<u8, &Arena>::with_capacity_in(128, &arena);
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
