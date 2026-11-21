# TripClipSkip Studio — current technical status

**Current runtime candidate: WP-03B1c-a-r1 (21 Nov 2026), read-only CPU preview; live GPU acceptance pending.** This documentation-sync package does **not** change runtime code, UI, native SDXL conditioning, or pooled-G. The underlying extension version string remains `0.3.16-conditioning-inspector-v2-preview`; the WP label identifies the engineering revision.

Native Proof now partially observes A1111 token chunks, job-local HMAC token-input fingerprints (no raw token IDs or text persisted), TI fix metadata, layer invocation links, first-chunk pooled object identity and construction boundaries. The independent r1 re-audit fixed cross-encoder L/G context inheritance for nested wrapper calls. **These observations do not establish final token/pooled provenance after custom `ASSEMBLE/POOL/REGION` operations.** Layer Composer injection is not enabled.

**Verification:** 758 CPU tests passed with the user's A1111 dev and custom source trees (2 pre-existing SyntaxWarnings); 736 passed / 22 skipped without external sources. Historical B1b-r1 SDXL GPU SHA/PNG evidence does **not** validate B1b-r2 or B1c-a-r1. The r3-r2 normalization-specific GPU gate also remains outstanding.

- [Russian README](README-RU.md) · [Implementation plan (RU)](PLAN-NEXT-RU.md) · [Roadmap](ROADMAP.md)
- [Current handoff (RU)](development/AI_HANDOFF_READ_FIRST_RU.md) · [Machine-readable status](development/CURRENT_STATE.json)
- [Independent r1 re-audit](development/WP03B1C_A_R1_INDEPENDENT_REAUDIT_RU.md) · [Docs synchronization](development/WP03B1C_A_R1_DOCSYNC_RU.md) · [Changelog](CHANGELOG.md)
