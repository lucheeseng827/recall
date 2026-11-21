//! `AdaptiveThreshold` — a feedback-driven [`ThresholdPolicy`] that targets an operator-chosen
//! **false-hit rate** instead of a fixed cosine cutoff (PLAN.md §5). A drop-in behind
//! the existing seam: the cache calls `decide` on the hot path and `observe` off it, exactly as for
//! `StaticThreshold`.
//!
//! The core idea (PLAN.md §5.4, "FHR is authoritative"): track the distribution of similarity scores
//! that were *confirmed wrong* (a served hit later judged a false hit). The threshold sits a
//! `z(target_fhr)`-sigma margin above that distribution's mean, so only a fraction ≈ `target_fhr` of
//! future false-hit-like scores can clear it. With no feedback yet, τ rests at a per-embedder
//! cold-start prior and the policy behaves like a static cutoff — so it is safe to enable before a
//! ground-truth feedback source exists (PLAN.md §5.5).
//!
//! Scope of this increment: a single adaptive band per namespace (per-region bands and the richer
//! per-query gates of §5.3 are a later seam extension), in-memory state, learning exercised by
//! direct `observe` calls. Wiring a production feedback source (a verifier) is the documented next
//! step (§5.4).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

use recall_core::{Outcome, ThresholdPolicy, Verdict};

use crate::moments::EwMoments;
use crate::stats::{ema, inv_upper_normal};

/// Tunables for the adaptive threshold. Construct with [`AdaptiveConfig::new`] so the safety clamps
/// are derived from the cold-start prior.
#[derive(Clone, Debug)]
pub struct AdaptiveConfig {
    /// The operator's SLO: the fraction of served hits allowed to be wrong. An auditable target
    /// ("≤2% wrong cache hits") rather than a magic cosine number. Clamped to `[1e-4, 0.5]`.
    pub target_false_hit_rate: f64,
    /// τ before any feedback — a per-embedder prior (PLAN.md §5.5). Clamped to `[0, 1]`.
    pub cold_start_tau: f32,
    /// Hard floor / ceiling on the learned τ so even adversarial feedback can't drive it to nonsense.
    pub tau_min: f32,
    pub tau_max: f32,
    /// Confirmed-wrong samples required before the Gaussian retune engages; below it, τ holds near
    /// the cold-start prior.
    pub min_samples: u64,
    /// EMA rate at which τ moves toward the freshly-computed FHR target.
    pub learn_rate: f64,
    /// Decay rate of the per-distribution moments.
    pub decay: f64,
}

impl AdaptiveConfig {
    /// Build a config from the two knobs an operator actually sets — the cold-start prior and the
    /// target false-hit rate — deriving the safety clamps (`tau_min = τ₀ − 0.1`, `tau_max = 0.995`,
    /// PLAN.md §5.5) and sensible learning defaults.
    pub fn new(cold_start_tau: f32, target_false_hit_rate: f64) -> Self {
        let tau0 = cold_start_tau.clamp(0.0, 1.0);
        Self {
            target_false_hit_rate: target_false_hit_rate.clamp(1e-4, 0.5),
            cold_start_tau: tau0,
            tau_min: (tau0 - 0.1).max(0.0),
            tau_max: 0.995,
            min_samples: 30,
            learn_rate: 0.1,
            decay: 0.05,
        }
    }
}

impl Default for AdaptiveConfig {
    /// A reasonable static-embedder prior: τ₀ = 0.85, target FHR = 2%.
    fn default() -> Self {
        Self::new(0.85, 0.02)
    }
}

/// One namespace's learned state.
struct NsState {
    /// Monotonic stamp of the last time this band was *used* — an observation folded in, or a
    /// lookup decided against it. An `AtomicU64` so the decision path can stamp it while holding
    /// only the **read** lock: making `decide` take the write lock to record recency would turn the
    /// hot path into a serialisation point, which is a worse problem than the one this solves.
    last_used: AtomicU64,
    tau: f32,
    /// Similarities that proved WRONG (false hits) — drives the FHR-targeting τ.
    neg: EwMoments,
    /// Similarities that proved RIGHT — used only as a recall guard so τ isn't raised above the
    /// proven-good cluster.
    pos: EwMoments,
}

/// A namespace heard from that does not yet hold a band.
///
/// Deliberately tiny — a count and a recency stamp, no moments and no τ. That is the whole design:
/// admission has to be cheap enough to hold an order of magnitude more namespaces than the band
/// budget, so a flood has to churn through a queue it cannot realistically outrun before a
/// legitimate namespace manages to post a second time.
struct Probationer {
    observations: u64,
    last_used: u64,
}

#[derive(Default)]
struct State {
    /// Namespaces holding learned state. Bounded by [`AdaptiveThreshold::MAX_NAMESPACES`].
    bands: HashMap<String, NsState>,
    /// The doorkeeper. Bounded by [`AdaptiveThreshold::PROBATION_NAMESPACES`].
    probation: HashMap<String, Probationer>,
}

/// A `ThresholdPolicy` whose per-namespace τ adapts toward a target false-hit rate as feedback
/// arrives. Interior-mutable (`decide` is `&self` on the hot path; `observe` is `&self` off it), so
/// it drops into the cache exactly where `StaticThreshold` did.
pub struct AdaptiveThreshold {
    cfg: AdaptiveConfig,
    id: String,
    state: RwLock<State>,
    /// Source of the recency stamps. A counter, not a clock: recency only ever needs to be
    /// *ordered*, and a counter cannot go backwards the way a wall clock can.
    clock: AtomicU64,
    evicted: AtomicU64,
}

impl AdaptiveThreshold {
    /// The most namespaces that can hold **learned state** at once. Matches the per-namespace
    /// savings ledger's bound, for the same reason: a namespace is caller-derived, and on the raw
    /// sidecar it is taken **verbatim from the request body** of `POST /v1/cache/feedback` — the
    /// one call that reaches `observe` from the network. Unbounded, a client could grow this map
    /// one entry per distinct namespace, for as long as it kept posting.
    pub const MAX_NAMESPACES: usize = 512;

    /// How many namespaces the doorkeeper remembers having heard from.
    ///
    /// Eight times the band budget, which it can afford because a [`Probationer`] is two integers
    /// and a key rather than two sets of moments. That ratio is the fix for the churn problem a
    /// small probation segment has: a namespace posting feedback slowly needs its first observation
    /// to still be remembered when its second arrives, and a flood evicts one probationer per post,
    /// so what matters is how many posts the flood must land between the two. A queue eight times
    /// the band budget buys eight times the grace.
    pub const PROBATION_NAMESPACES: usize = 4096;

    /// Observations that earn a band. Two, not one: a single observation is exactly what a
    /// namespace that posted once and never again looks like, and telling those apart from real
    /// traffic is the doorkeeper's entire job.
    pub const PROMOTE_AFTER: u64 = 2;

    pub fn new(cfg: AdaptiveConfig) -> Self {
        let id = format!(
            "adaptive@fhr={:.3}:tau0={:.3}",
            cfg.target_false_hit_rate, cfg.cold_start_tau
        );
        Self {
            cfg,
            id,
            state: RwLock::new(State::default()),
            clock: AtomicU64::new(0),
            evicted: AtomicU64::new(0),
        }
    }

    fn tick(&self) -> u64 {
        self.clock.fetch_add(1, Ordering::Relaxed)
    }

    /// Whether a namespace holds a learned band. Exposed for tests and operator introspection;
    /// does **not** count as a use, so asking cannot change what gets evicted.
    pub fn is_tracked(&self, ns: &str) -> bool {
        self.state.read().unwrap().bands.contains_key(ns)
    }

    /// Whether the doorkeeper remembers a namespace that has not yet earned a band.
    pub fn is_on_probation(&self, ns: &str) -> bool {
        self.state.read().unwrap().probation.contains_key(ns)
    }

    /// How many namespaces hold learned state. At most [`Self::MAX_NAMESPACES`].
    pub fn tracked_namespaces(&self) -> usize {
        self.state.read().unwrap().bands.len()
    }

    /// How many namespaces the doorkeeper is holding. At most [`Self::PROBATION_NAMESPACES`].
    pub fn probation_namespaces(&self) -> usize {
        self.state.read().unwrap().probation.len()
    }

    /// The current learned τ for a namespace, or the cold-start prior if it has no band.
    /// Exposed for tests and operator introspection — deliberately **not** a use, so reading a
    /// band cannot save it from eviction. [`Self::tau_touch`] is the decision path's version.
    pub fn tau(&self, ns: &str) -> f32 {
        self.state
            .read()
            .unwrap()
            .bands
            .get(ns)
            .map_or(self.cfg.cold_start_tau, |s| s.tau)
    }

    /// τ for a decision, marking the band as used. A namespace serving live traffic keeps its
    /// learned band even while no feedback arrives for it — otherwise the cutoff under production
    /// traffic could be reset to the cold-start prior by an unrelated namespace needing room.
    ///
    /// Reads `bands` only. A namespace still on probation has learned nothing yet, so there is
    /// nothing for traffic to protect, and checking the second map would put a lookup the hot path
    /// cannot use on every cache miss.
    fn tau_touch(&self, ns: &str) -> f32 {
        let stamp = self.tick();
        let st = self.state.read().unwrap();
        match st.bands.get(ns) {
            Some(s) => {
                s.last_used.store(stamp, Ordering::Relaxed);
                s.tau
            }
            None => self.cfg.cold_start_tau,
        }
    }
}

impl ThresholdPolicy for AdaptiveThreshold {
    fn id(&self) -> &str {
        &self.id
    }

    fn decide(&self, ns: &str, top: Option<f32>) -> Verdict {
        match top {
            Some(score) if score >= self.tau_touch(ns) => Verdict::Hit,
            _ => Verdict::Miss,
        }
    }

    fn observe(&self, ns: &str, score: f32, outcome: Outcome) {
        let stamp = self.tick();
        let mut st = self.state.write().unwrap();

        // A namespace with a band just learns.
        if let Some(band) = st.bands.get_mut(ns) {
            band.fold(stamp, score, outcome, &self.cfg);
            return;
        }

        // Otherwise it goes through the doorkeeper, which never allocates a band and never touches
        // one. A namespace that posts once and never again therefore costs two integers and a key
        // until it ages out — and costs a band nothing at all, however many of them arrive.
        //
        // Bands are never merged into a shared overflow row the way the savings ledger merges its
        // own. That row only accounts, so merging there costs resolution and nothing else. A band
        // *decides* — τ is the cutoff a future lookup is measured against — and pooling distinct
        // namespaces' score distributions would let traffic that is not a namespace's own move the
        // threshold its hits are served at.
        let earned = knock(&mut st.probation, ns, stamp);
        if earned < Self::PROMOTE_AFTER {
            return;
        }
        if !self.open_band(&mut st, ns, earned, stamp) {
            return;
        }
        // The band starts from this observation. The one that earned entry is not carried over:
        // holding moments for every probationer is exactly the cost the doorkeeper exists to avoid,
        // and `retune` does not engage until `min_samples` anyway, so one sample is nothing.
        if let Some(band) = st.bands.get_mut(ns) {
            band.fold(stamp, score, outcome, &self.cfg);
        }
    }

    fn namespaces_evicted(&self) -> u64 {
        self.evicted.load(Ordering::Relaxed)
    }
}

/// Record one knock at the door, admitting the namespace to the queue if it is new (evicting the
/// least recently heard-from probationer when the queue is full). Returns what it has now earned.
fn knock(probation: &mut HashMap<String, Probationer>, ns: &str, stamp: u64) -> u64 {
    if !probation.contains_key(ns) && probation.len() >= AdaptiveThreshold::PROBATION_NAMESPACES {
        if let Some(victim) = lru_key(probation.iter().map(|(k, p)| (k, p.last_used))) {
            probation.remove(&victim);
        }
    }
    let p = probation.entry(ns.to_string()).or_insert(Probationer {
        observations: 0,
        last_used: stamp,
    });
    p.last_used = stamp;
    p.observations += 1;
    p.observations
}

/// The least recently used key of an `(key, stamp)` sequence.
///
/// A linear scan, because this runs only when a bounded map is full. An O(1) LRU wants an intrusive
/// list or a second index to keep in sync; neither is worth the bug surface off the path the module
/// docs already describe as "off the hot path".
fn lru_key<'a>(it: impl Iterator<Item = (&'a String, u64)>) -> Option<String> {
    it.min_by_key(|(_, stamp)| *stamp).map(|(k, _)| k.clone())
}

/// Total observations folded into a band, of either outcome. What "this namespace is real" is
/// measured in.
fn observations(s: &NsState) -> u64 {
    s.neg.count() + s.pos.count()
}

impl AdaptiveThreshold {
    /// Give a namespace that has knocked enough times a band, if it can have one.
    ///
    /// With room, that is an insert. Full, it is an **admission decision**, and a comparative one:
    /// the candidate must have knocked at least as many times as the band it would displace has
    /// been observed. Without that comparison the doorkeeper is only a speed bump — a flood pays
    /// two posts per namespace and starts evicting bands with thousands of observations behind
    /// them. With it, taking a band costs as much feedback as that band has ever received.
    ///
    /// The displaced namespace returns to the queue **carrying what it had earned**, so it is one
    /// observation away from winning its slot back rather than starting over. Its moments do not
    /// survive — the queue holds no moments, which is what makes it affordable — so this counts as
    /// an eviction: learned state was lost.
    fn open_band(&self, st: &mut State, ns: &str, earned: u64, stamp: u64) -> bool {
        if st.bands.len() >= Self::MAX_NAMESPACES {
            let Some(victim) = lru_key(
                st.bands
                    .iter()
                    .map(|(k, s)| (k, s.last_used.load(Ordering::Relaxed))),
            ) else {
                return false;
            };
            let theirs = st.bands.get(&victim).map_or(0, observations);
            if earned < theirs {
                // Not yet worth a slot. It stays in the queue and tries again on its next
                // observation, so a namespace that keeps earning feedback is never turned away for
                // good.
                return false;
            }
            st.bands.remove(&victim);
            self.evicted.fetch_add(1, Ordering::Relaxed);
            let displaced = knock(&mut st.probation, &victim, stamp);
            if let Some(p) = st.probation.get_mut(&victim) {
                p.observations = theirs.max(displaced);
            }
        }
        st.probation.remove(ns);
        st.bands.insert(
            ns.to_string(),
            NsState {
                last_used: AtomicU64::new(stamp),
                tau: self.cfg.cold_start_tau,
                neg: EwMoments::new(self.cfg.decay),
                pos: EwMoments::new(self.cfg.decay),
            },
        );
        true
    }
}

impl NsState {
    /// Fold one observation in and retune, marking the band as used.
    fn fold(&mut self, stamp: u64, score: f32, outcome: Outcome, cfg: &AdaptiveConfig) {
        self.last_used.store(stamp, Ordering::Relaxed);
        match outcome {
            Outcome::Wrong => self.neg.update(score as f64),
            Outcome::Agree => self.pos.update(score as f64),
        }
        retune(self, cfg);
    }
}

/// Recompute τ from the confirmed-wrong distribution. FHR is authoritative (PLAN.md §5.4): place τ a
/// `z(target_fhr)`-sigma margin above the mean of similarities that proved wrong, so ≈ `target_fhr`
/// of that distribution still clears τ and the rest is rejected. Until enough wrong samples exist,
/// hold τ near the cold-start prior. A recall guard keeps τ from rising above the proven-good
/// cluster (don't start rejecting answers feedback has confirmed correct).
fn retune(state: &mut NsState, cfg: &AdaptiveConfig) {
    if state.neg.count() < cfg.min_samples {
        state.tau = ema(state.tau as f64, cfg.cold_start_tau as f64, 0.05) as f32;
        return;
    }
    let z = inv_upper_normal(cfg.target_false_hit_rate);
    let mut target = state.neg.mean() + z * state.neg.std();
    if state.pos.count() >= cfg.min_samples {
        // Recall guard: don't raise τ *into* the proven-correct cluster — capping at its mean would
        // reject roughly half the correct hits (those below the mean). Cap at a lower-bound
        // statistic (mean − 2σ) so the bulk of the positive cluster still clears τ.
        let pos_floor = state.pos.mean() - 2.0 * state.pos.std();
        target = target.min(pos_floor);
    }
    let next = ema(state.tau as f64, target, cfg.learn_rate) as f32;
    state.tau = next.clamp(cfg.tau_min, cfg.tau_max);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> AdaptiveThreshold {
        AdaptiveThreshold::new(AdaptiveConfig::new(0.85, 0.02))
    }

    #[test]
    fn cold_start_behaves_like_a_static_cutoff() {
        let p = policy();
        assert_eq!(p.tau("ns"), 0.85);
        assert!(matches!(p.decide("ns", Some(0.90)), Verdict::Hit));
        assert!(matches!(p.decide("ns", Some(0.80)), Verdict::Miss));
        assert!(matches!(p.decide("ns", None), Verdict::Miss));
    }

    #[test]
    fn false_hit_feedback_raises_tau_and_makes_the_cache_more_precise() {
        let p = policy();
        // A spread of confirmed-WRONG hits centered ~0.85 — the cache was over-serving here.
        for i in 0..120 {
            let score = 0.80 + (i % 11) as f32 * 0.01; // 0.80..=0.90, mean ≈ 0.85
            p.observe("ns", score, Outcome::Wrong);
        }
        // τ climbs a margin above the false-hit mean (precision up): a 0.85 candidate that hit at
        // cold-start now misses, while a clearly-similar 0.97 candidate still hits.
        assert!(
            p.tau("ns") > 0.88,
            "tau rose above the false-hit band, got {}",
            p.tau("ns")
        );
        assert!(p.tau("ns") <= 0.995, "respects the ceiling");
        assert!(matches!(p.decide("ns", Some(0.85)), Verdict::Miss));
        assert!(matches!(p.decide("ns", Some(0.97)), Verdict::Hit));
    }

    #[test]
    fn learning_is_per_namespace() {
        let p = policy();
        for i in 0..120 {
            p.observe("hot", 0.80 + (i % 11) as f32 * 0.01, Outcome::Wrong);
        }
        // The untouched namespace still decides at the cold-start prior.
        assert_eq!(p.tau("other"), 0.85);
        assert!(matches!(p.decide("other", Some(0.86)), Verdict::Hit));
        // While the namespace that saw false hits is now stricter.
        assert!(matches!(p.decide("hot", Some(0.86)), Verdict::Miss));
    }

    #[test]
    fn agree_feedback_does_not_raise_tau() {
        let p = policy();
        for _ in 0..120 {
            p.observe("ns", 0.95, Outcome::Agree);
        }
        // Confirmed-correct hits are not false hits — τ must not climb on them.
        assert!(
            p.tau("ns") <= 0.85 + 1e-6,
            "agree feedback keeps tau at the prior"
        );
    }

    #[test]
    fn recall_guard_does_not_cut_into_the_correct_cluster() {
        let p = policy();
        // Confirmed-WRONG hits sit high (~0.95) — the FHR target alone would push τ up near there.
        // Confirmed-CORRECT hits form a spread cluster around ~0.90 (0.86..=0.94). The recall guard
        // must keep τ below the *lower edge* of that cluster, not at its mean.
        for i in 0..80 {
            p.observe("ns", 0.93 + (i % 5) as f32 * 0.01, Outcome::Wrong); // 0.93..=0.97
            p.observe("ns", 0.86 + (i % 9) as f32 * 0.01, Outcome::Agree); // 0.86..=0.94, mean ≈ 0.90
        }
        // τ is pulled below the positive mean — the old `min(pos.mean())` guard would have parked it
        // at ≈0.90 and rejected the lower half of the correct cluster.
        assert!(
            p.tau("ns") < 0.90,
            "guard keeps tau below the positive mean, got {}",
            p.tau("ns")
        );
        // A correct hit below the positive mean still serves.
        assert!(matches!(p.decide("ns", Some(0.88)), Verdict::Hit));
    }
}

#[cfg(test)]
mod bound_tests {
    use super::*;

    const MAX: usize = AdaptiveThreshold::MAX_NAMESPACES;
    const QUEUE: usize = AdaptiveThreshold::PROBATION_NAMESPACES;
    const PROMOTE: u64 = AdaptiveThreshold::PROMOTE_AFTER;

    fn policy() -> AdaptiveThreshold {
        AdaptiveThreshold::new(AdaptiveConfig::new(0.85, 0.02))
    }

    fn feed(p: &AdaptiveThreshold, ns: &str, n: u64) {
        for _ in 0..n {
            p.observe(ns, 0.99, Outcome::Wrong);
        }
    }

    /// Post `n` distinct namespaces, `each` observations apiece — what a flood looks like.
    fn flood(p: &AdaptiveThreshold, tag: &str, n: usize, each: u64) {
        for i in 0..n {
            feed(p, &format!("{tag}-{i}"), each);
        }
    }

    /// `observe` is reachable from the network in exactly one place — `POST /v1/cache/feedback`,
    /// whose namespace comes verbatim from the request body. Unbounded, each distinct namespace
    /// bought a permanent entry, so a client could grow this state for as long as it kept posting.
    #[test]
    fn both_maps_are_bounded_however_many_namespaces_arrive() {
        let p = policy();
        flood(&p, "ns", QUEUE + 2_000, 1);
        assert!(p.probation_namespaces() <= QUEUE);
        assert_eq!(
            p.tracked_namespaces(),
            0,
            "nothing posted twice, so the flood never allocated a single band"
        );
        assert_eq!(p.namespaces_evicted(), 0, "and never cost one either");
    }

    /// Two knocks earn a band. One is exactly what a namespace that posted once and never again
    /// looks like, and telling those apart is the doorkeeper's whole job.
    #[test]
    fn a_band_is_opened_once_a_namespace_has_knocked_more_than_once() {
        let p = policy();
        feed(&p, "real", 1);
        assert!(p.is_on_probation("real"));
        assert!(!p.is_tracked("real"), "one observation proves nothing");

        feed(&p, "real", 1);
        assert!(p.is_tracked("real"), "the second observation earns a band");
        assert!(!p.is_on_probation("real"), "and it leaves the queue");
    }

    /// **The churn edge.** A namespace posting feedback slowly needs its first observation to still
    /// be remembered when its second arrives. A flood evicts one probationer per post, so what
    /// matters is how many posts it can land in between — and the queue is eight times the band
    /// budget precisely to make that number large. At the old 128-slot probation segment this
    /// namespace was gone long before its second post.
    #[test]
    fn a_slow_namespace_still_earns_a_band_through_a_sustained_flood() {
        let p = policy();
        feed(&p, "slow", 1);
        // An absolute number, not one derived from the constant under test: 2000 posts is far past
        // the 128-slot probation segment this replaces and comfortably inside the new queue, so
        // shrinking the queue back fails this test instead of quietly rescaling it.
        flood(&p, "junk", 2_000, 1);
        assert!(
            p.is_on_probation("slow"),
            "the first knock is still remembered after 2000 intervening namespaces"
        );

        feed(&p, "slow", 1);
        assert!(
            p.is_tracked("slow"),
            "so the second one earns a band, which is what the churn edge denied"
        );
    }

    /// And the honest limit of that: a flood longer than the queue does evict a single knock. The
    /// cost is one observation — the namespace knocks again next time and starts over — never a
    /// band, because a one-knock namespace has none.
    #[test]
    fn a_flood_longer_than_the_queue_costs_a_single_knock_and_never_a_band() {
        let p = policy();
        feed(&p, "slow", 1);
        flood(&p, "junk", QUEUE + 10, 1);
        assert!(!p.is_on_probation("slow"), "the knock did age out");
        assert_eq!(p.namespaces_evicted(), 0, "but no band was ever at risk");

        feed(&p, "slow", PROMOTE);
        assert!(p.is_tracked("slow"), "and knocking again still works");
    }

    /// A band is unreachable from the queue. However long a one-shot flood runs, it allocates
    /// nothing and displaces nothing.
    #[test]
    fn a_barely_started_band_survives_a_flood_that_would_have_evicted_it() {
        let p = policy();
        feed(&p, "new-team", PROMOTE);
        assert!(p.is_tracked("new-team"));

        flood(&p, "junk", QUEUE * 2, 1);

        assert!(p.is_tracked("new-team"), "the band is untouched");
        assert_eq!(
            p.tracked_namespaces(),
            1,
            "and no band was opened by the flood"
        );
        assert_eq!(p.namespaces_evicted(), 0);
    }

    /// Paying the promotion price reaches the door, so admission itself has to be comparative —
    /// otherwise the flood simply posts twice per namespace and evicts trained bands anyway.
    #[test]
    fn a_flood_that_pays_the_promotion_price_still_cannot_displace_a_trained_band() {
        let p = policy();
        for i in 0..MAX {
            feed(&p, &format!("real-{i}"), 5);
        }
        assert_eq!(p.tracked_namespaces(), MAX);

        flood(&p, "junk", MAX * 4, PROMOTE);

        assert_eq!(p.namespaces_evicted(), 0, "no band was displaced");
        for i in 0..MAX {
            assert!(
                p.is_tracked(&format!("real-{i}")),
                "real-{i} was displaced by a 2-observation newcomer"
            );
        }
    }

    /// The comparison must not become a new lockout. A namespace that keeps earning has to be able
    /// to take a slot from one that earned less — and the loser returns to the queue carrying what
    /// it earned, one knock from winning back rather than starting over.
    #[test]
    fn a_namespace_that_out_knocks_a_band_takes_its_place_and_it_returns_to_the_queue() {
        let p = policy();
        for i in 0..MAX {
            feed(&p, &format!("modest-{i}"), 3);
        }
        assert_eq!(p.tracked_namespaces(), MAX);

        feed(&p, "earner", 20);

        assert!(p.is_tracked("earner"), "sustained feedback earned a band");
        assert_eq!(
            p.tracked_namespaces(),
            MAX,
            "the budget is still exactly full"
        );
        assert_eq!(
            p.namespaces_evicted(),
            1,
            "displacing a band is an eviction"
        );
        assert!(
            p.is_on_probation("modest-0"),
            "the displaced namespace is back in the queue, not forgotten"
        );
        assert!(!p.is_tracked("modest-0"));
    }

    /// The point of displacing rather than refusing: a namespace that shows up after the budget is
    /// spent must still be able to learn. Under first-come-first-served it never could.
    #[test]
    fn a_namespace_that_arrives_late_is_admitted_rather_than_locked_out() {
        let p = policy();
        for i in 0..MAX {
            feed(&p, &format!("early-{i}"), 3);
        }
        assert!(!p.is_tracked("latecomer"));

        feed(&p, "latecomer", p.cfg.min_samples + 5);
        assert!(p.is_tracked("latecomer"), "the newcomer got in");
        assert_ne!(
            p.tau("latecomer"),
            p.cfg.cold_start_tau,
            "and actually learned, rather than holding the prior"
        );
        assert!(p.tracked_namespaces() <= MAX, "still bounded");
    }

    /// Recency counts lookups, not just feedback — a namespace serving production traffic with no
    /// feedback source attached must not lose the cutoff its hits are served at. `decide` records
    /// that without ever taking the write lock the hot path cannot afford.
    #[test]
    fn serving_traffic_keeps_a_band_alive_with_no_feedback_arriving() {
        let p = policy();
        for i in 0..MAX {
            feed(&p, &format!("ns-{i}"), 3);
        }
        // `ns-0` is the least recently *observed*, so it is next to be displaced — unless being
        // decided against counts, which is the point.
        assert!(matches!(p.decide("ns-0", Some(0.99)), Verdict::Hit));

        feed(&p, "earner", 20);
        assert_eq!(p.namespaces_evicted(), 1);
        assert!(p.is_tracked("ns-0"), "the band being served survived");
        assert!(!p.is_tracked("ns-1"), "the genuinely idle one went instead");
    }

    /// Introspection is not use. Reading a band to report on it must not change which band is
    /// evicted next, or a metrics scrape would perturb the thing it is measuring.
    #[test]
    fn reading_a_band_does_not_save_it() {
        let p = policy();
        for i in 0..MAX {
            feed(&p, &format!("ns-{i}"), 3);
        }
        let _ = p.tau("ns-0");
        let _ = p.is_tracked("ns-0");
        let _ = p.tracked_namespaces();

        feed(&p, "earner", 20);
        assert!(
            !p.is_tracked("ns-0"),
            "the least recently *used* band still went, whatever was read"
        );
    }
}
