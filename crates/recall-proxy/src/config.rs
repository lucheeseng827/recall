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
        }
    }
}
