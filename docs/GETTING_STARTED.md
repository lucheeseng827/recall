# Getting started with recall

A 10-minute tour: prove the cache loop works on your machine, then run recall as a
drop-in caching proxy in front of an (mock) OpenAI-compatible upstream and watch the
same request go from **~1.2 s (upstream)** to **~2 ms (cache hit)** — including across
a restart.

Everything below was captured from real runs. All you need is a Rust toolchain
([rustup](https://rustup.rs)) and, for the proxy demo, Python 3 (only to fake a slow
upstream — any OpenAI- or Anthropic-compatible endpoint works instead).

## 1. The 60-second proof: `recall bench`

From the repo root:

```bash
cargo run -p recall -- bench
```

`bench` warms the cache with 5 entries, then fires 2000 lookups (half exact repeats,
half paraphrases) and reports the hot-path numbers:

```text
recall bench  (embedder: hash-v1@256, policy: static@0.850)
  entries warmed  : 5
  iterations      : 2000
  hit-rate        : 50.0%  (1000/2000)
  embed  p50/p99  : 11.0 / 29.9 µs   (query → vector; embedder-bound)
  lookup p50/p99  : 23.2 / 53.8 µs   (full get: shortcut → embed → ANN → decide)
  index  p50      : ~12.2 µs   (lookup_p50 − embed_p50: in-process ANN + threshold)
  lookup max      : 109.2 µs
```

A full cache lookup — hash-embed the query, search the index, apply the threshold —
costs ~23 µs. The 50% hit-rate is expected here: the default build uses a
deterministic *hash* embedder (zero ML dependencies), which only matches exact text.
Paraphrase hits come from building with `--features static` and pointing `--model` at
a local [model2vec/potion](https://github.com/MinishLab/model2vec-rs) directory — see
[CONFIG.md](./CONFIG.md).

## 2. Drive the loop by hand: `recall ask`

```bash
echo "Use the 'Forgot password' link on the login page." | \
  cargo run -p recall -- ask --ns demo "how do I reset my password"
```

```text
MISS — reading completion from stdin to store…
MISS (stored 49 bytes)
```

A **miss** hands you the query vector and stores whatever you feed it; a repeat of the
same prompt inside one process is a **hit**. (`ask` is deliberately a single-shot,
in-memory proof of the loop — persistence is the proxy's job, next.)

## 3. The real thing: `recall serve` as a caching proxy

### 3a. Fake a slow upstream (skip if you have a real one)

Save as `mock_upstream.py` — it answers every chat completion after a deliberate
1.2 s delay, so the cache win is visible:

```python
import json, time
from http.server import BaseHTTPRequestHandler, HTTPServer

REPLY = {
    "id": "chatcmpl-mock-1", "object": "chat.completion",
    "created": 1700000000, "model": "mock-gpt",
    "choices": [{"index": 0, "message": {"role": "assistant",
        "content": "To reset your password, use the 'Forgot password' link on the login page."},
        "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 17, "total_tokens": 29},
}

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        time.sleep(1.2)  # pretend to be a slow LLM
        body = json.dumps(REPLY).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a): pass

print("mock upstream on :9099")
HTTPServer(("127.0.0.1", 9099), Handler).serve_forever()
```

```bash
python mock_upstream.py &
```

### 3b. Start the proxy with a durable store

```bash
cargo build -p recall --features store-redb

RECALL_UPSTREAM=http://127.0.0.1:9099 \
  ./target/debug/recall serve --listen 127.0.0.1:8080 \
  --store redb --db-path demo.redb
```

(Against a real upstream instead: set `RECALL_UPSTREAM` to its base URL and export
`RECALL_UPSTREAM_API_KEY` — keys are env-only, never flags, never logged.)

### 3c. Same request, twice

```bash
REQ='{"model":"mock-gpt","messages":[{"role":"user","content":"How do I reset my password?"}]}'
curl -si -w '\ntime_total: %{time_total}s\n' \
  http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' -d "$REQ" | grep -iE '^HTTP|x-recall|time_total'
```

First call — a **miss**: recall forwards to the upstream, stores the reply, and you
pay the full upstream latency:

```text
HTTP/1.1 200 OK
x-recall-cache: miss
time_total: 1.216472s
```

Run the exact same `curl` again — a **hit**: the upstream is never called.

```text
HTTP/1.1 200 OK
x-recall-cache: hit
x-recall-namespace: default:openai:mock-gpt:9d61185a31c85d5d
x-recall-score: 1
time_total: 0.001701s
```

**1.216 s → 0.0017 s.** The two extra headers are exactly what you feed back to
`POST /v1/cache/feedback` to train the adaptive threshold — no client-side key
reconstruction needed (see [API.md](./API.md)).

### 3d. It's observable

```bash
curl -s http://127.0.0.1:8080/metrics | grep -E 'recall_(requests|hits|misses|tokens_saved|hit_ratio)_?'
```

```text
recall_requests_total 2
recall_hits_total 1
recall_misses_total 1
recall_tokens_saved_total 29
recall_hit_ratio 0.5000
```

Latency histograms (`recall_cache_get_duration_seconds`,
`recall_upstream_duration_seconds`) are in the same scrape.

### 3e. Hits survive a restart

Kill the proxy (Ctrl-C) and start it again with the same `--db-path`. On startup it
**rehydrates** — re-derives each persisted entry's vector and rebuilds the in-memory
index — and logs it (run with `RUST_LOG=info`):

```text
INFO durable redb store — rehydrated cached lookups across restart db_path="demo.redb" rehydrated=1 persisted=1
```

The same request is still a hit, with the upstream still untouched:

```text
HTTP/1.1 200 OK
x-recall-cache: hit
time_total: 0.001959s
```

## 4. Where to go next

- **Every flag / env / TOML key**: [CONFIG.md](./CONFIG.md) — including
  `--policy adaptive --target-fhr 0.02` (a learned, per-namespace threshold instead
  of one magic cutoff), `--index hnsw` for large corpora (validate with
  `recall ann-bench` first), and `--watch-config` hot-reload.
- **All endpoints, headers, hit semantics**: [API.md](./API.md) — Anthropic
  `/v1/messages` is cached too, and streaming (SSE) responses are cached and
  replayed as streams.
- **Deploy, backup, monitoring, security posture**: [OPERATIONS.md](./OPERATIONS.md).
  Read the security section before exposing anything beyond loopback.
- **Use it as a library** (no proxy hop): the `SemanticCache` example in the
  [README](../README.md#use-it-as-a-library).
- **Reproduce the numbers**: `cargo run -p recall-eval` (threshold-policy comparison)
  and `cargo bench --workspace` (criterion hot-path + index crossover).
