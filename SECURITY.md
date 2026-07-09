# Security Policy

## Reporting a Vulnerability

Please report security issues **privately**. Do not open a public issue for a
suspected vulnerability.

- Preferred: open a private [GitHub Security Advisory](https://github.com/lucheeseng827/recall/security/advisories/new)
  with details and reproduction steps.
- If you cannot use Security Advisories, contact the maintainer via their GitHub
  profile ([@lucheeseng827](https://github.com/lucheeseng827)) to arrange a private channel.

We aim to acknowledge reports within a few business days and follow a 90-day
coordinated disclosure window, crediting reporters who wish to be named.

## Supported Versions

Until 1.0, the latest published `0.x` minor is supported. From 1.0 onward we
support the latest minor and the previous one.

| Version | Supported |
|---------|-----------|
| latest `0.x` | ✅ |
| older   | ❌ |

## Security model — read before you deploy

recall's open-source build is designed to run **inside your own trust boundary**
(embedded as a library, or as a localhost/sidecar proxy). Two properties an
operator must understand:

- **Namespace isolation is structural, not authenticated.** Partitions are
  isolated by construction — you cannot reach another partition's entries — but
  the cache does **not** authenticate tenancy. If you derive a tenant from a
  client-supplied header without verifying it, a client can name another tenant's
  namespace. **Do not deploy recall as a multi-tenant *trust* boundary** —
  authenticated, hard multi-tenant isolation is not something this project
  provides.

- **The proxy and `/metrics` endpoints are unauthenticated and bind to loopback
  by default.** The raw cache sidecar routes (`/v1/cache/{insert,feedback}`) write
  to the cache; keep them on loopback or a trusted network and do not expose them
  to untrusted callers. Upstream API keys are read from the environment
  (`RECALL_UPSTREAM_API_KEY`, `RECALL_ANTHROPIC_API_KEY`) only, never from a flag,
  and are never logged.

## Cache poisoning

A cache miss populates from the upstream response, so only point recall at a
trusted upstream. The cache key includes the model and answer-affecting decode
parameters, and the namespace is keyed by tenant + model, which bounds the blast
radius of any single populated entry.
