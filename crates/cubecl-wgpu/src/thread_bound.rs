//! Thread affinity for wgpu's WebGPU objects on wasm with atomics.
//!
//! On `wasm32` with the `atomics` target feature, wgpu's browser backend hands
//! out JS values. A JS value is an index into the heap table of the thread that
//! made it, so it means nothing (or something else) on any other thread, and
//! wgpu marks those types `!Send` there. The runtime keeps its server in a
//! registry that needs `Send`, so this module makes the server types `Send` and
//! puts the check where the types cannot hold it:
//!
//! - `cubecl-common`'s device registry records the thread that first locates a
//!   device and panics when another thread reaches the device. Every access to
//!   the server goes through that registry.
//! - [`ThreadBound`] wraps the values that leave the registry on their own: the
//!   futures a read or a sync returns, and the cached instances. It checks the
//!   thread on every access and leaks its value if another thread drops it.
//!
//! On every other target nothing here has any effect: [`ThreadBound`] is a
//! plain wrapper, and the `unsafe impl`s are not compiled.

use core::future::Future;
use core::mem::ManuallyDrop;
use core::pin::Pin;
use core::task::{Context, Poll};

/// A value that can only be used on the thread that wrapped it, on wasm with
/// atomics. A plain wrapper everywhere else.
pub(crate) struct ThreadBound<T> {
    value: ManuallyDrop<T>,
    #[cfg(all(target_family = "wasm", target_feature = "atomics"))]
    owner: std::thread::ThreadId,
}

impl<T> ThreadBound<T> {
    /// Binds `value` to the current thread.
    pub(crate) fn new(value: T) -> Self {
        Self {
            value: ManuallyDrop::new(value),
            #[cfg(all(target_family = "wasm", target_feature = "atomics"))]
            owner: std::thread::current().id(),
        }
    }

    /// Whether the current thread may use the value.
    #[inline]
    pub(crate) fn is_local(&self) -> bool {
        #[cfg(all(target_family = "wasm", target_feature = "atomics"))]
        {
            std::thread::current().id() == self.owner
        }
        #[cfg(not(all(target_family = "wasm", target_feature = "atomics")))]
        {
            true
        }
    }

    #[inline]
    #[track_caller]
    fn assert_local(&self) {
        assert!(
            self.is_local(),
            "a wgpu object was used on a thread other than the one that made it; on wasm \
             with atomics a device and its objects stay on one thread"
        );
    }

    /// The value. Panics on a thread other than the owner.
    #[track_caller]
    pub(crate) fn get(&self) -> &T {
        self.assert_local();
        &self.value
    }
}

impl<T> Drop for ThreadBound<T> {
    fn drop(&mut self) {
        if self.is_local() {
            // SAFETY: `value` is dropped once, here, and never touched again.
            unsafe { ManuallyDrop::drop(&mut self.value) };
        } else {
            // Dropping a JS value on another thread would release a slot in
            // that thread's heap table. Leaking it is the only safe choice.
            log::warn!("wgpu: leaked an object dropped on a thread other than its own");
        }
    }
}

impl<F: Future> Future for ThreadBound<F> {
    type Output = F::Output;

    #[track_caller]
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.assert_local();
        // SAFETY: `value` is structurally pinned. It is never moved out of
        // `self`, and `Drop` only drops it in place.
        let future = unsafe { self.map_unchecked_mut(|this| &mut *this.value) };
        future.poll(cx)
    }
}

// SAFETY: every access to the value checks the owner thread first, and a drop
// on another thread leaks the value. The value itself never reaches a thread it
// does not belong to.
#[cfg(all(target_family = "wasm", target_feature = "atomics"))]
unsafe impl<T> Send for ThreadBound<T> {}
#[cfg(all(target_family = "wasm", target_feature = "atomics"))]
unsafe impl<T> Sync for ThreadBound<T> {}

// SAFETY (all four): these types live inside a device's server, which
// `cubecl-common`'s registry only lets the device's owner thread reach on wasm
// with atomics. A `WgpuResource` that a caller takes out with `get_resource`
// must stay on that thread too, and the docs on `WgpuResource` say so.
#[cfg(all(target_family = "wasm", target_feature = "atomics"))]
mod server_types {
    use crate::{WgpuCompiler, WgpuResource, WgpuServer, WgpuStorage, compute::copies::WgpuCopies};

    unsafe impl<C: WgpuCompiler> Send for WgpuServer<C> {}
    unsafe impl Send for WgpuStorage {}
    unsafe impl Send for WgpuResource {}
    unsafe impl Send for WgpuCopies {}
}
