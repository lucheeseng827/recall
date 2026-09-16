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

## 4. Measure it on your own traffic — before you commit

Everything above proves the loop works. It does not tell you what recall is worth
*to you* — that depends entirely on how repetitive your traffic actually is.
`recall replay` answers that from a log of requests you already have, with nothing
in your production request path.

Give it a JSONL file, one request body per line — the same bodies you send your
provider today:

```text
{"model":"mock-gpt","temperature":0,"messages":[{"role":"user","content":"How do I reset my password?"}]}
{"model":"mock-gpt","temperature":0,"messages":[{"role":"user","content":"What is your refund policy?"}]}
...
```

(Lines may also be `{"path":"/v1/messages","body":{…}}` to mix Anthropic traffic into
the same file.)

To get dollars rather than tokens, give the proxy your rates — see
[`[pricing.<model>]`](./CONFIG.md#pricingmodel--report-savings-in-dollars). Copy them
off a provider invoice; recall ships no price list and never looks one up:

```toml
# recall.toml
listen = "127.0.0.1:8080"
upstream = "http://127.0.0.1:9099"

[pricing."mock-gpt"]
input = 2.50      # USD per 1M input (prompt) tokens
output = 10.00    # USD per 1M output (completion) tokens
```

Start the proxy with that config, then replay the log at it:

```bash
./target/debug/recall serve --config recall.toml &
./target/debug/recall replay --file traffic.jsonl --target http://127.0.0.1:8080
```

```text
recall replay  (target: http://127.0.0.1:8080, 200 lines)
  $ saved         : $0.0392   (priced per model by the target's [pricing] table)
                    not discounted for wrong hits — add --verify-sample to measure them
  hit-rate        : 98.0%  (196 of 200 cacheable)
  tokens saved    : 5684 total  (input 2352, output 3332)
  requests        : 200  (hits 196, misses 4, bypass 0, errors 0)
```

> **Replaying against the mock upstream?** Make it keep-alive-capable first, or you
> will see spurious `errors` where the connection was dropped between requests: add
> `protocol_version = "HTTP/1.1"` to the `Handler` class and swap `HTTPServer` for
> `ThreadingHTTPServer` (imported from the same module). One `curl` at a time does
> not need this; 200 pooled requests do. A real upstream already handles it.

Reading it:

- **`$ saved`** is priced **per model**, by the proxy, from your table — a log spanning
  several models is priced with each model's own rates, and the input/output split is
  respected (output typically costs 3–5× input). With no table configured the line
  says so instead of printing `$0.00`; `--price-input`/`--price-output` give a flat
  blended estimate as a fallback.
- **The hit-rate is the honest one for a cold start.** The first request for each
  distinct question misses and fills the cache — 4 misses above, one per distinct
  question — exactly as a fresh deployment would behave. Replaying the same log a
  second time reports the *warm* number; both are real, they answer different
  questions. Each run reports only itself, so a second replay is not the first one
  counted twice.
- **`$ saved` is not discounted for wrong hits** unless you measure them. Add
  `--verify-sample 0.1 --upstream <url>` and recall re-asks the upstream for 10% of
  its hits, compares the answers, and prints the savings **both** gross and net of
  the measured disagreement rate. Only meaningful at `temperature: 0` — a sampled
  model disagrees with itself. A wrong hit is not a saving, so quote the discounted
  number to anyone who will hold you to it.

If the hit-rate here is low, your traffic is not repetitive enough for a semantic
cache to pay — and you found that out from a log, in a few minutes, without a
deployment.

## 5. Check in on it: `recall status`

Once it is running, one command summarizes what the cache has actually done —
the screen to paste into a thread when someone asks whether it is worth keeping:

```bash
./target/debug/recall status            # --target http://host:port for a remote one
```

```text
recall status  (target: http://127.0.0.1:8095)
  $ saved         : $0.0394
                    $0.0381 less the 3.3% observed false-hit rate
  hit-rate        : 98.0%  (197 of 201 cacheable)
  tokens saved    : 5713 total  (input 2364, output 3349)
  entries         : 4 held by the store
  requests        : 201  (hits 197, misses 4, bypass 0, upstream errors 0)
  false-hit       : 3.3% observed  (2 of 60 judged hits reported wrong)
```

The second line is the one that matters when you quote this to anyone. recall
discounts the savings by the **observed** false-hit rate — the share of served hits
someone actually reported wrong through
[`POST /v1/cache/feedback`](./API.md#post-v1cachefeedback--train-the-adaptive-threshold). Report outcomes and you
get a number you can defend; report none and the line says plainly that the figure
is **not** discounted, because nobody looking is not the same as nobody finding
anything.

It also refuses to round its caveats away. Saved tokens on a model with no price
rule, hits whose stored completion carried no usage at all, and streamed misses
that were dropped rather than cached each get their own line when they occur — all
three make the headline figure an **under-count**, and a number that is quietly too
low costs trust as fast as one that is too high. A clean deployment prints no
caveats at all, so seeing one always means something.

`--json` emits the raw snapshot instead, for your own tooling.

## 6. Where to go next

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
- **Every `replay` flag**: [CONFIG.md](./CONFIG.md#recall-replay--drive-a-running-serve-with-a-request-log).
