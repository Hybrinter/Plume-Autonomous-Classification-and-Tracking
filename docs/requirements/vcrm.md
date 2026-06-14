# PACT Verification Cross-Reference Matrix (VCRM)

> **Source of truth:** `docs/requirements/vcrm.toml`. This table is rendered from it.
> CI (`scripts/check_vcrm.py`) enforces that every running-venue requirement is both cited by a
> module docstring (`Satisfies:`) and backed by evidence, and that no PIL/HIL requirement claims
> `verified`.

## Scope (thin slice)

This matrix covers only requirements exercised by the two **running** validation profiles:

| Profile | File | Venue |
| --- | --- | --- |
| SIL (full sim) | `profiles/sil.toml` | `sil` |
| SIL + real link | `profiles/sil-link-real.toml` | `sil-link-real` |

PIL and HIL profiles are **DEFINED-NOT-RUN** (no hardware yet); requirements verifiable only there
are deliberately absent rather than falsely marked verified.

## Matrix

| Requirement | Statement | Method | Venue | Evidence | Status |
| --- | --- | --- | --- | --- | --- |
| REQ-COMM-HIGH-003 | Authenticated command ingress (HMAC + accepted sources) | SIL | sil | test_iss_ingress_pipeline; scenario:ingress_auth_accept | verified |
| REQ-COMM-HIGH-004 | Command ACK/NACK for every ingested command | SIL | sil | test_iss_iface_app; scenario:ingress_nack_bad_hmac | verified |
| REQ-COMM-HIGH-002 | CCSDS framing + CRC integrity | SIL | sil | test_ccsds_codec; scenario:downlink_ccsds_frames | verified |
| REQ-COMM-HIGH-001 | Downlink gated by AOS visibility | SIL | sil-link-real | test_sil_closed_loop; scenario:aos_los_gating | verified |
| REQ-SAFE-HIGH-002 | Thermal over-limit -> SAFE + stow | SIL | sil | test_sil_closed_loop; scenario:safe_on_thermal | verified |
| REQ-AIML-GIMB-001 | Autonomous closed-loop pointing toward plume | SIL | sil | test_sil_closed_loop; scenario:closed_loop_pointing | verified |
| REQ-GIMB-HIGH-001 | ROI retention within pointing deadband | SIL | sil | test_sil_closed_loop; scenario:closed_loop_pointing | verified |
| REQ-GIMB-HIGH-003 | Runaway gimbal detection forces stow | SIL | sil | test_runaway; scenario:safe_on_thermal | verified |

## Permanent gaps

| Gap | Statement | Status |
| --- | --- | --- |
| GAP-GROUND-SEGMENT | Real ground segment is never tested; the GSE station emulator stands in for it. The `lock` axis (LaunchLock) is likewise a permanent VCRM gap -- there is no device and no config field, only this record. | gap |
