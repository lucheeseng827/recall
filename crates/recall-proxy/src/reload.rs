//! Hot-swappable threshold policy (PLAN.md §3-OSS "Config + hot-reload", §7). The cache holds a
//! [`SwappablePolicy`] in place of a fixed `Box<dyn ThresholdPolicy>`; a [`PolicyHandle`] shared with
//! the config watcher lets the running threshold be replaced at runtime — `arc-swap` so `decide`
//! reads the current snapshot lock-free on the hot path and a swap never blocks a lookup.
//!
//! Only the **policy** is swappable. The listener, index, store, and embedder own bound sockets and
//! loaded weights/file handles and are *not* hot-reloadable — the binary rejects a config change to
//! those with a clear log line and a restart hint (PLAN.md §3-OSS). The metrics registry lives on
//! `ProxyState`, outside the swap, so counters survive a reload.
//!
//! `policy.id()` is **not** folded into the cache key (only `embedder.id()` is — see
//! `recall_core::SemanticCache::ns_key`), so swapping the policy never invalidates or shifts cached
//! entries.

use std::sync::Arc;

use arc_swap::ArcSwap;

use recall_core::{Outcome, ThresholdPolicy, Verdict};

/// Shared, atomically-swappable threshold policy. The cache's [`SwappablePolicy`] and the config
/// watcher both hold an `Arc<PolicyHandle>`; the watcher calls [`store`](PolicyHandle::store) to swap
/// in a freshly-parsed policy while the hot path keeps reading the current one lock-free.
///
/// The snapshot is `Arc<Box<dyn ThresholdPolicy>>` — arc-swap stores a sized `Arc<T>`, and the boxed
/// trait object is that sized `T`; the double indirection is one extra pointer hop, off any hot loop.
pub struct PolicyHandle {
    current: ArcSwap<Box<dyn ThresholdPolicy>>,
}

impl PolicyHandle {
    pub fn new(initial: Box<dyn ThresholdPolicy>) -> Arc<Self> {
        Arc::new(Self {
            current: ArcSwap::from_pointee(initial),
        })
    }

    /// Replace the active policy. Lock-free; a concurrent `decide`/`observe` either sees the old or
    /// the new snapshot, never a torn state.
    pub fn store(&self, policy: Box<dyn ThresholdPolicy>) {
        self.current.store(Arc::new(policy));
    }

    /// The active policy's id (owned) — used to log a reload (`old → new`) and to report the live
    /// policy where an owned string is acceptable.
    pub fn current_id(&self) -> String {
        self.current.load().id().to_string()
    }

    fn decide(&self, ns: &str, top: Option<f32>) -> Verdict {
        self.current.load().decide(ns, top)
    }

    fn observe(&self, ns: &str, score: f32, outcome: Outcome) {
        self.current.load().observe(ns, score, outcome);
    }
}

/// A `ThresholdPolicy` that delegates every decision to whatever policy its [`PolicyHandle`] currently
/// holds — so the cache can be built once and have its threshold hot-swapped underneath it.
///
/// `id()` returns a fixed label captured at construction (the trait returns `&str`, which can't borrow
/// through the arc-swap snapshot). The *live* id after a reload is reported in the reload log line and
/// via [`PolicyHandle::current_id`]; decisions always use the current policy regardless.
pub struct SwappablePolicy {
    handle: Arc<PolicyHandle>,
    id: String,
}

impl SwappablePolicy {
    pub fn new(handle: Arc<PolicyHandle>) -> Self {
        let id = format!("swappable({})", handle.current_id());
        Self { handle, id }
    }
}

impl ThresholdPolicy for SwappablePolicy {
    fn id(&self) -> &str {
        &self.id
    }
    fn decide(&self, ns: &str, top: Option<f32>) -> Verdict {
        self.handle.decide(ns, top)
    }
    fn observe(&self, ns: &str, score: f32, outcome: Outcome) {
        self.handle.observe(ns, score, outcome);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use recall_core::StaticThreshold;

    #[test]
    fn swapping_the_handle_changes_decisions_live() {
        let handle = PolicyHandle::new(Box::new(StaticThreshold::new(0.9)));
        let policy = SwappablePolicy::new(handle.clone());

        // At τ=0.9 a 0.85 neighbour misses.
        assert!(matches!(policy.decide("ns", Some(0.85)), Verdict::Miss));

        // Hot-swap to a more permissive τ=0.8 — the SAME policy object now serves it.
        handle.store(Box::new(StaticThreshold::new(0.8)));
        assert!(matches!(policy.decide("ns", Some(0.85)), Verdict::Hit));

        // The reported live id reflects the swap (the trait `id()` stays the construction-time label).
        assert_eq!(handle.current_id(), "static@0.800");
        assert!(policy.id().starts_with("swappable(static@0.900)"));
    }
}
