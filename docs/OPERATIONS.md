# recall — operations runbook

Deploy, state & backup, upgrade, monitoring, troubleshooting, security. Grounded in
the code as built. Knobs live in [CONFIG.md](./CONFIG.md), endpoints in
[API.md](./API.md), deployment topologies (single node vs partitioned fleet) in the
[README §Architecture](../README.md#deployment-topology). Reproduce measured numbers
with `recall bench`, `recall ann-bench`, and `cargo bench --workspace`.

## Deploy

recall is one static binary; a deployment is one `recall serve` process per cache
shard. Production shape:

```sh
# Build with the production seams (real embedder + durable store):
cargo build -p recall --release --features static,store-redb

# Run (systemd/container); keys are env-only, never flags:
export RECALL_UPSTREAM_API_KEY=sk-...
export RECALL_ANTHROPIC_API_KEY=sk-ant-...
recall serve --config recall.toml \
  --model /var/lib/recall/potion-base-8M \
  --store redb --db-path /var/lib/recall/cache.redb \
  --index hnsw --policy adaptive --target-fhr 0.02
```

Expected startup log (real output; `rehydrated`/`persisted` grow with the store):

```text
INFO durable redb store — rehydrated cached lookups across restart db_path="/var/lib/recall/cache.redb" rehydrated=0 persisted=0
INFO recall serve: listening (cache-miss forwarding) listen=127.0.0.1:8080 openai_upstream=https://api.openai.com anthropic_upstream=https://api.anthropic.com
```

Placement rules (the one fact that shapes everything: **the index lives
in-process**, and redb is **single-writer**):

- Exactly one live `serve` per `--db-path` volume.
- Fleet scale = **routing, not replication**: consistent-hash the LB on namespace so
  a namespace always lands on the same instance (details + trade-offs in the README
  topology section). The LB must pass SSE through unbuffered and use `/healthz`.
- Route `/v1/cache/feedback` with the same affinity, so the adaptive threshold
  trains on the instance that served the hit.
- A `--model` directory must be present locally (offline load; no download at boot).

## State & backup

| what | where | loss impact |
|---|---|---|
| Cache entries (prompt + completion blobs) | the redb file at `--db-path` (with `--store redb`) | cold cache: correctness unaffected (every miss just goes upstream), hit-rate and token savings reset |
| In-memory ANN index + exact-map | process RAM, **rebuilt from redb at startup** (`rehydrate`) | rebuilt automatically; nothing to back up |
| Adaptive-threshold state (per-namespace moments) | process RAM only | resets to the cold-start τ₀ on restart; retrains from feedback |
| Metrics counters | process RAM only | reset on restart (rate()-based alerting is immune) |

Backup: the cache is *derived data* — the honest posture is that losing it costs
money (upstream re-buys), not correctness. If the warm cache is worth backing up,
copy the redb file **while the instance is stopped** (single-writer; there is no
online-snapshot hook), or snapshot the volume. Restore = put the file back, start
`serve`, and let rehydration rebuild the index; the startup log's
`rehydrated N / persisted M` confirms it.

The memory store (`--store memory`, the default) has no state to manage at all.

## Upgrade / rollback

- Binary upgrades with `--store memory` are trivial (the cache was ephemeral anyway).
- With `--store redb`: the entries are plain blobs and the schema is stable today,
  but there is **no `format_version` yet** — keep a copy of the redb file before a
  version jump, and treat a downgrade the same way.
- **Never change the embedder (`--model`, or hash ↔ static) against an existing
  redb file** and expect the same hits: rehydration re-embeds persisted prompts
  under the *current* embedder, so similarity geometry shifts. Changing embedders
  is a re-key: start a fresh `--db-path` (or accept a effectively-cold cache) and
  re-run `recall calibrate` for the new τ.
- τ / policy / target-FHR changes need no restart with `--watch-config`
  (threshold-policy-only hot reload; see [CONFIG.md](./CONFIG.md#hot-reload---watch-config)).
- Rollback = previous binary + the pre-upgrade redb copy.

## Monitoring

Scrape `GET /metrics` per instance (full metric table in
[API.md](./API.md#get-healthz-and-get-metrics)). What to alert on:

| signal | alert when | meaning / action |
|---|---|---|
| `/healthz` | non-200 | instance down; LB should already have ejected it |
| `recall_hit_ratio` | sustained drop | traffic shifted (new namespaces are cold), τ too strict after a reload, or an embedder/DB mismatch after a deploy |
| `rate(recall_upstream_errors_total)` | > 0 sustained | upstream unreachable/refusing — callers are getting 502s on misses |
| `rate(recall_stream_not_stored_total)` | non-zero rate | upstream SSE streams ending dirty (timeouts, `max_tokens` truncation, disconnects) — those replies are never cached, so identical requests keep missing |
| `rate(recall_bypass_total)` | unexpectedly high share | callers sending tools / `n>1` / high temperature / `anthropic-beta` headers — cacheable traffic is lower than you think |
| `recall_cache_get_duration_seconds` | p99 above your SLO | index too big for `brute` (consider `--index hnsw`, but validate recall first — see troubleshooting), or a slow embedder |
| false-hit rate | above `--target-fhr` | measure with `recall replay --verify-sample`; raise τ or feed more `/v1/cache/feedback` |

Logs: `RUST_LOG=info` for startup/reload/warnings; `RUST_LOG=recall=debug` for
per-request hit/miss/bypass events (structural fields only — prompts and
completions are never logged, by design).

Savings validation: `recall replay --file reqs.jsonl --price-input … --price-output …`
against a running instance reports hit-rate, the input/output token split, and a
dollar estimate; `--verify-sample 0.1` re-asks the upstream for 10% of hits to
sample the candidate false-hit rate.

## Troubleshooting (symptom-first)

**Every response says `x-recall-cache: bypass`.** The requests carry a bypass
trigger: `tools`/`functions`, `n > 1` (OpenAI), `temperature` above
`--max-temperature`, or (Anthropic) an `anthropic-beta` header / non-default
`anthropic-version`. `RUST_LOG=recall=debug` logs the reason per request. Raise
`--max-temperature` only if replaying sampled completions is acceptable for you.

**Requests keep missing that "should" hit.**
- Different `model` or any answer-affecting decode param (incl. `stream` and
  `stream_options`) → a different namespace by design; check `x-recall-namespace`
  on a hit to see the derived key.
- Paraphrases miss with the default **hash embedder** — it is a deterministic test
  stub, not a semantic model. Build with `--features static` and set `--model`
  (the startup log must say `loaded static embedder from …`; if it says
  `--model needs a --features static build`, the flag was ignored).
- τ too strict for your embedder: run `recall calibrate --file pairs.jsonl` and set
  the recommended τ.

**Wrong answers are being replayed (false hits).** Lower is not the only lever:
raise `--tau`, or enable `--policy adaptive --target-fhr …` and report offenders
via `POST /v1/cache/feedback` with `outcome: "wrong"` (each one raises that
namespace's cutoff). Quantify first with `recall replay --verify-sample`.

**`serve` exits immediately with code 2.** Config rejected — the message says
which: `--store redb requires a --features store-redb build`, `unknown --store/--index/--policy '…'`,
`invalid <config file>`, `--watch-config requires --config`, or a redb open failure
(`--store redb: open <path> failed`). recall refuses to silently downgrade
durability rather than start with less than you asked for.

**Cache is cold after every restart.** You are on `--store memory` (the default) —
use `--store redb --db-path …`. If you *are* on redb, check the startup log:
`rehydrated=0 persisted=0` means the db file is new/empty (wrong path?);
`rehydration failed; serving with a cold index` means the scan errored — the blobs
are intact and entries re-learn on first miss, but investigate the logged error.

**`--store redb: open … failed` at startup.** The file is locked by another live
instance (redb is single-writer — one `serve` per file) or permissions/volume are
wrong. On Windows dev boxes a stray `recall.exe` holding the file also blocks
rebuilds — stop it first.

**Streaming responses never get cached.** Check `recall_stream_not_stored_total`:
the upstream streams are ending without a clean terminator (`[DONE]` /
`message_stop`) or with a truncation finish reason — recall refuses to cache a
partial reply. Also confirm your LB isn't buffering/cutting SSE.

**Clients hang or 502 on misses.** The upstream is slow/down; forwards are bounded
by the internal connect (10 s) / request (120 s) timeouts and surface as 502 plus
`recall_upstream_errors_total`. Hits keep being served from cache regardless — only
misses depend on the upstream.

**Config edits do nothing.** Hot reload covers only `policy`/`tau`/`target_fhr`,
needs `--watch-config --config <file>`, and a field pinned by a flag/env at startup
keeps winning (precedence is preserved across reloads). Everything else needs a
restart — the log says `config field changed but is not hot-reloadable`. A TOML
typo logs `config reload skipped: parse error` and keeps the previous policy.

**Feedback returns 400.** `outcome` must be exactly `agree` or `wrong`, and `score`
a finite cosine similarity in [-1, 1] (use the `x-recall-score` header / lookup
`score` verbatim).

**Latency grows with cache size.** `--index brute` is exact but O(N) per lookup —
it becomes the bottleneck past a few thousand entries. `--index hnsw` removes it
(14–25× faster at 50k) **but validate recall first**: on random 256-dim vectors
(the bundled static embedder's width) HNSW recall@1 collapses at 10k+ entries with
the current tuning — run `ann-bench` for your dims/N before opting in.

## Security posture

What listens, what authenticates, what is deliberately out of scope (the disclosure
policy is [SECURITY.md](../SECURITY.md); the trust-boundary statement is in the
[README §Security model](../README.md#security-model)).

- **One listener**: `recall serve` on `--listen` (default `127.0.0.1:8080`), plain
  HTTP, **no authentication of callers** — anyone who can reach it can read cached
  completions via lookups, poison the cache via `/v1/cache/insert`, skew the
  adaptive threshold via `/v1/cache/feedback`, and spend your upstream key via
  misses. Keep it loopback/VPC-internal, or front it with your own authenticating
  proxy. TLS is likewise the fronting layer's job.
- **Namespace isolation is structural, not authenticated**: nothing verifies
  that a caller may use a namespace. Do not deploy recall as a multi-tenant
  *trust* boundary — it does not provide authenticated isolation.
- **Secrets**: upstream keys are env-only (`RECALL_UPSTREAM_API_KEY`,
  `RECALL_ANTHROPIC_API_KEY`) — never flags, never in the TOML, never logged. A
  caller-supplied auth header is forwarded to the upstream as-is and never stored.
- **Data at rest**: cached prompts/completions are **plaintext** in RAM and in the
  redb file. Apply disk encryption and file permissions per your data class; cached
  LLM traffic is customer content.
- **Logging redaction**: prompts and completion bodies are never written to logs —
  only namespace, score, status, latency (enforced in `recall-proxy`).
- **Egress**: the only network calls are to the two configured upstreams on miss/
  bypass; the default build has no other network dependency (air-gap capable — the
  static embedder loads locally).
