//! The operator's price book: turns saved tokens into saved dollars, locally.
//!
//! `recall_tokens_saved_total` is the honest unit of what the cache did; it is not the unit anyone
//! budgets in. This module is the (optional) `[pricing.<model>]` table from `recall.toml` that maps
//! a model id to what the operator actually pays for it, so `/metrics` can also export
//! `recall_cost_saved_usd_total` (PLAN.md §3-OSS).
//!
//! Three rules, because a savings figure nobody trusts is worse than no savings figure at all:
//!
//! 1. **The operator's own numbers.** There is no bundled price list and no network lookup — vendor
//!    prices change, differ by contract, and would rot in the binary. An empty price book (the
//!    default) prices nothing, and that is a fine state to run in.
//! 2. **Never guess a model.** A model with no matching rule contributes exactly `$0` and is counted
//!    in `recall_unpriced_tokens_saved_total` — a visible flag, so a wrong price book reads as
//!    "unpriced", never as "we saved nothing". Prefix rules exist (`"gpt-4o*"`) but only where the
//!    operator wrote one deliberately; recall never infers a family from a model id.
//! 3. **Integer money.** Rates are converted to integer micro-USD per million tokens at load and all
//!    accumulation is integer (`u64` micro-USD, saturating). Floats appear only at render.
//!
//! Input and output tokens are priced separately: providers charge output at roughly 3–5× input, so
//! pricing the lumped total mis-states the savings whenever the mix is skewed.

use std::collections::HashMap;

use serde::Deserialize;

/// One model's rates, as written in `recall.toml` — **USD per million tokens**, the unit every
/// provider publishes, so an operator can copy the number off an invoice without converting it.
///
/// ```toml
/// [pricing."gpt-4o"]
/// input = 2.50
/// output = 10.00
/// ```
#[derive(Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct RateSpec {
    /// USD per 1M input (prompt) tokens.
    pub input: f64,
    /// USD per 1M output (completion) tokens.
    pub output: f64,
}

/// A loaded rate, in integer micro-USD per million tokens. Converting once at load keeps the hot
/// path integer-only and keeps a repeated `f64` accumulation from drifting over millions of hits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rate {
    input_micros_per_mtok: u64,
    output_micros_per_mtok: u64,
}

/// How a price-book key matches a model id. Matching is **case-sensitive** either way — the model
/// id is compared byte-for-byte as the caller sent it, so a mismatched-case key prices nothing and
/// its tokens surface as unpriced rather than silently matching the wrong rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Match {
    /// `"gpt-4o"` — matches that model id exactly.
    Exact,
    /// `"gpt-4o*"` — matches any model id starting with the stem, so an operator can cover dated
    /// snapshots (`gpt-4o-2024-08-06`) with one rule. Opt-in per entry: a bare key never prefixes.
    Prefix,
}

/// The operator's model → rate table. Empty by default; an empty book prices nothing and flags
/// every model as unpriced rather than inventing a number.
#[derive(Clone, Debug, Default)]
pub struct PriceBook {
    /// `(stem, how-it-matches, rate)`. Small (an operator writes a handful of entries) and read on
    /// the hit path only, so a linear scan beats a map once prefix rules are in play.
    rules: Vec<(String, Match, Rate)>,
}

/// USD per million tokens → micro-USD per million tokens. Negative, NaN, and infinite rates are
/// rejected as `None` (a nonsense rate must not silently become `$0` savings, nor a huge one).
fn to_micros_per_mtok(usd: f64) -> Option<u64> {
    if !usd.is_finite() || usd < 0.0 {
        return None;
    }
    let micros = (usd * 1_000_000.0).round();
    // Guard the cast: an absurd rate would wrap or saturate meaninglessly.
    if micros > u64::MAX as f64 {
        return None;
    }
    Some(micros as u64)
}

impl PriceBook {
    /// Build a price book from the parsed `[pricing.*]` table.
    ///
    /// Returns the book plus the keys that were **rejected** (a negative/NaN/absurd rate, or an
    /// empty stem). Rejections are returned rather than silently dropped so `serve` can warn: a
    /// typo'd rate that quietly priced nothing would be indistinguishable from a working cache that
    /// saved nothing.
    pub fn load(table: &HashMap<String, RateSpec>) -> (Self, Vec<String>) {
        let mut rules = Vec::with_capacity(table.len());
        let mut rejected = Vec::new();
        for (key, spec) in table {
            let (stem, how) = match key.strip_suffix('*') {
                Some(stem) => (stem, Match::Prefix),
                None => (key.as_str(), Match::Exact),
            };
            match (
                stem.is_empty(),
                to_micros_per_mtok(spec.input),
                to_micros_per_mtok(spec.output),
            ) {
                (false, Some(input_micros_per_mtok), Some(output_micros_per_mtok)) => rules.push((
                    stem.to_string(),
                    how,
                    Rate {
                        input_micros_per_mtok,
                        output_micros_per_mtok,
                    },
                )),
                _ => rejected.push(key.clone()),
            }
        }
        // Longest stem first, so the most specific rule wins: with both `gpt-4o*` and
        // `gpt-4o-mini*` configured, `gpt-4o-mini-2024-07-18` must price as mini.
        rules.sort_by_key(|rule| std::cmp::Reverse(rule.0.len()));
        (Self { rules }, rejected)
    }

    /// True when no rule was configured — the default. `serve` uses this to say once, at startup,
    /// that the dollar figure will stay at zero, instead of letting an operator read an empty
    /// `recall_cost_saved_usd_total` as "the cache saved nothing".
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// The number of configured rules (exact + prefix).
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// The cost in **micro-USD** of `input`/`output` tokens for `model`, or `None` when no rule
    /// matches — the caller must then count the tokens as unpriced, never as zero-cost.
    ///
    /// Saturating throughout: a hostile or absurd token count clamps instead of wrapping into a
    /// small number, because a wrapped savings figure is worse than a clamped one.
    pub fn cost_micros(&self, model: &str, input: u64, output: u64) -> Option<u64> {
        let rate = self.rate_for(model)?;
        // micro-USD = tokens × (micro-USD per 1M tokens) / 1M. Multiply first (in u128, which cannot
        // overflow for u64 × u64) so sub-million token counts do not truncate to zero.
        let cost = |tokens: u64, per_mtok: u64| -> u64 {
            let micros = (tokens as u128 * per_mtok as u128) / 1_000_000;
            u64::try_from(micros).unwrap_or(u64::MAX)
        };
        Some(
            cost(input, rate.input_micros_per_mtok)
                .saturating_add(cost(output, rate.output_micros_per_mtok)),
        )
    }

    /// The most specific rule matching `model`, if any.
    fn rate_for(&self, model: &str) -> Option<Rate> {
        self.rules
            .iter()
            .find(|(stem, how, _)| match how {
                Match::Exact => model == stem,
                Match::Prefix => model.starts_with(stem.as_str()),
            })
            .map(|(_, _, rate)| *rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn book(entries: &[(&str, f64, f64)]) -> PriceBook {
        let table = entries
            .iter()
            .map(|(k, i, o)| {
                (
                    (*k).to_string(),
                    RateSpec {
                        input: *i,
                        output: *o,
                    },
                )
            })
            .collect();
        let (book, rejected) = PriceBook::load(&table);
        assert!(rejected.is_empty(), "unexpected rejections: {rejected:?}");
        book
    }

    #[test]
    fn prices_input_and_output_at_their_own_rates() {
        let b = book(&[("gpt-4o", 2.50, 10.00)]);
        // 1M input @ $2.50 + 1M output @ $10.00 = $12.50 = 12_500_000 micro-USD.
        assert_eq!(
            b.cost_micros("gpt-4o", 1_000_000, 1_000_000),
            Some(12_500_000)
        );
        // The split matters: pricing the 2M lumped total at either single rate would be wrong.
        assert_eq!(b.cost_micros("gpt-4o", 2_000_000, 0), Some(5_000_000));
        assert_eq!(b.cost_micros("gpt-4o", 0, 2_000_000), Some(20_000_000));
    }

    #[test]
    fn unknown_model_is_none_never_zero() {
        let b = book(&[("gpt-4o", 2.50, 10.00)]);
        // The distinction the whole module exists for: "no rule" is not "free".
        assert_eq!(b.cost_micros("claude-sonnet-4", 1_000, 1_000), None);
        // And an exact key does NOT silently cover a dated snapshot — that needs a prefix rule.
        assert_eq!(b.cost_micros("gpt-4o-2024-08-06", 1_000, 1_000), None);
    }

    #[test]
    fn prefix_rule_is_opt_in_and_most_specific_wins() {
        let b = book(&[("gpt-4o*", 2.50, 10.00), ("gpt-4o-mini*", 0.15, 0.60)]);
        assert_eq!(
            b.cost_micros("gpt-4o-2024-08-06", 1_000_000, 0),
            Some(2_500_000)
        );
        // `gpt-4o-mini-…` matches both stems; the longer (more specific) one must win.
        assert_eq!(
            b.cost_micros("gpt-4o-mini-2024-07-18", 1_000_000, 0),
            Some(150_000)
        );
    }

    #[test]
    fn small_token_counts_do_not_truncate_to_zero() {
        let b = book(&[("m", 2.50, 10.00)]);
        // 1_000 output tokens @ $10/Mtok = $0.01 = 10_000 micro-USD. Dividing before multiplying
        // (per-token rate first) would floor this to 0 and lose every small hit.
        assert_eq!(b.cost_micros("m", 0, 1_000), Some(10_000));
        // A single token is worth 10 micro-USD — still representable, still not zero.
        assert_eq!(b.cost_micros("m", 0, 1), Some(10));
    }

    #[test]
    fn nonsense_rates_are_rejected_not_silently_zero() {
        let table: HashMap<String, RateSpec> = [
            (
                "neg".to_string(),
                RateSpec {
                    input: -1.0,
                    output: 1.0,
                },
            ),
            (
                "nan".to_string(),
                RateSpec {
                    input: f64::NAN,
                    output: 1.0,
                },
            ),
            (
                "inf".to_string(),
                RateSpec {
                    input: 1.0,
                    output: f64::INFINITY,
                },
            ),
            (
                "".to_string(),
                RateSpec {
                    input: 1.0,
                    output: 1.0,
                },
            ),
            (
                "ok".to_string(),
                RateSpec {
                    input: 1.0,
                    output: 1.0,
                },
            ),
        ]
        .into_iter()
        .collect();
        let (b, mut rejected) = PriceBook::load(&table);
        rejected.sort();
        assert_eq!(rejected, vec!["", "inf", "nan", "neg"]);
        assert_eq!(b.len(), 1);
        // A rejected key prices nothing — it reports as unpriced, exactly like an absent model.
        assert_eq!(b.cost_micros("neg", 1_000_000, 0), None);
        assert_eq!(b.cost_micros("ok", 1_000_000, 0), Some(1_000_000));
    }

    #[test]
    fn empty_book_prices_nothing() {
        let b = PriceBook::default();
        assert!(b.is_empty());
        assert_eq!(b.cost_micros("gpt-4o", 1_000_000, 1_000_000), None);
    }

    #[test]
    fn absurd_token_counts_clamp_instead_of_wrapping() {
        let b = book(&[("m", 1000.0, 1000.0)]);
        // u64::MAX tokens at $1000/Mtok overflows u64 micro-USD; it must clamp, not wrap to a
        // small, plausible-looking number.
        assert_eq!(b.cost_micros("m", u64::MAX, u64::MAX), Some(u64::MAX));
    }
}
