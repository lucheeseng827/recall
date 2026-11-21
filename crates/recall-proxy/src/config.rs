//! Proxy configuration. Deliberately small for M1; durable-backend and per-namespace knobs arrive
//! with the OSS bar (PLAN.md §3-OSS). The proxy fronts both OpenAI-shaped (`/v1/chat/completions`)
//! and Anthropic-shaped (`/v1/messages`) traffic, each with its own upstream + auth.

/// Runtime configuration for the dual OpenAI/Anthropic-compatible proxy.
#[derive(Clone, Debug)]
pub struct Config {
    /// Address the proxy listens on, e.g. `127.0.0.1:8080`.
    pub listen: String,
    /// Upstream OpenAI-compatible base URL (no trailing `/v1`), e.g. `https://api.openai.com`.
    pub upstream_base: String,
    /// Bearer key used upstream when the incoming request carries no `Authorization` header. When the
    /// caller sends their own `Authorization`, it is forwarded as-is and this is unused.
    pub upstream_api_key: Option<String>,
    /// The outermost cache namespace component; tenancy (when present) is layered on top of this.
    pub base_namespace: String,
    /// Requests with `temperature` strictly above this **bypass** the cache (lookup *and* store):
    /// replaying a highly-sampled completion is a silent behavior change (PLAN.md §3-OSS cache-key
    /// rule). OpenAI's and Anthropic's default temperature is 1.0, so the default here caches typical
    /// traffic while still letting an operator dial it down for strict determinism.
    pub max_temperature: f64,
    /// Upstream **connect** timeout (seconds): caps how long a single TCP/TLS connection attempt to
    /// the upstream may take before it is treated as an error.
    pub connect_timeout_secs: u64,
    /// Upstream **request** timeout (seconds): the end-to-end ceiling for a forwarded call. Generous
    /// by default because LLM completions are slow, but finite so a hung upstream can't pin a worker.
    pub request_timeout_secs: u64,
    /// Upstream Anthropic Messages base URL (no trailing `/v1`), e.g. `https://api.anthropic.com`.
    /// Used for `/v1/messages` traffic; OpenAI traffic still goes to `upstream_base`.
    pub anthropic_upstream_base: String,
    /// `anthropic-version` header sent upstream when the caller omits one (Anthropic requires it).
    pub anthropic_version: String,
    /// `x-api-key` used upstream when the incoming `/v1/messages` request carries none. When the
    /// caller sends their own `x-api-key`, it is forwarded as-is and this is unused. Env-only.
    pub anthropic_api_key: Option<String>,
    /// The operator's model → rate table (`[pricing.<model>]` in `recall.toml`), used to turn saved
    /// tokens into the `recall_cost_saved_usd_total` metric. Empty by default: recall ships no price
    /// list and makes no network call, so the dollar figure is always the operator's own numbers or
    /// nothing at all (see [`crate::pricing`]). Not hot-reloadable — a rate change needs a restart.
    /// OTLP/HTTP base for `gen_ai.*` span export (e.g. `http://collector:4318`). `None` — the
    /// default — means no exporter is built and no span leaves the process, so the network-free
    /// default stands. See [`crate::otel`].
    pub otel_endpoint: Option<String>,
    /// `service.name` on exported spans. Only read when `otel_endpoint` is set.
    pub otel_service_name: String,
    pub pricing: crate::pricing::PriceBook,
    /// What the store was at start-up: which backend, and how much of a previous run's cache it
    /// recovered. Facts, not knobs — the binary fills them in after opening the store, and
    /// `/metrics` reports them so "did my cache survive the restart" is a ratio of two measured
    /// numbers rather than a log line someone has to have been watching for.
    pub store: StoreFacts,
}

/// What the store was when this process started.
///
/// Reported as-is and never editorialized: recall says which backend is in use and how many entries
/// it recovered, and stops there. A node cannot observe what any other node holds, so it never
/// claims anything about sharing — that arithmetic belongs to whatever is scraping every node,
/// which genuinely can see them all.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoreFacts {
    /// Backend name as the operator selected it (`memory`, `redb`).
    pub backend: &'static str,
    /// Whether that backend survives a restart at all. `false` for `memory`, where an empty cache
    /// at start-up is the design rather than a loss.
    pub durable: bool,
    /// Rows the store held before rehydration ran — the denominator of "recovered N of M".
    pub persisted_at_start: u64,
    /// Entries rebuilt into the live index and exact-map at start-up: cache this process did **not**
    /// have to buy again. Always 0 for an ephemeral backend, which has nothing to rebuild.
    ///
    /// Lower than `persisted_at_start` is expected, not an error — rows written under a different
    /// embedder cannot be reused, because a different embedder re-keys similarity entirely.
    pub rehydrated: u64,
}

impl Default for StoreFacts {
    fn default() -> Self {
        Self {
            backend: "memory",
            durable: false,
            persisted_at_start: 0,
            rehydrated: 0,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            listen: "127.0.0.1:8080".to_string(),
            upstream_base: "https://api.openai.com".to_string(),
            upstream_api_key: None,
            base_namespace: "default".to_string(),
            max_temperature: 1.0,
            connect_timeout_secs: 10,
            request_timeout_secs: 120,
            anthropic_upstream_base: "https://api.anthropic.com".to_string(),
            anthropic_version: "2023-06-01".to_string(),
            anthropic_api_key: None,
            // No exporter by default: the network-free default is the product's argument, not a
            // convenience, so telemetry is opted into rather than out of.
            otel_endpoint: None,
            otel_service_name: "recall".to_string(),
            pricing: crate::pricing::PriceBook::default(),
            store: StoreFacts::default(),
        }
    }
}
