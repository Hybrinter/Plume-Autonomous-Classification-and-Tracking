# PACT Verification Cross-Reference Matrix (VCRM)

> **Source of truth:** `docs/requirements/vcrm.toml`. This table is rendered from it.
> CI (`scripts/check_vcrm.py`) enforces that every running-venue requirement is both cited by a
> module docstring (`Satisfies:`) and backed by evidence, and that no PIL/HIL requirement claims
> `verified`.

## Scope (thin slice)

This matrix covers requirements exercised by the **running** venues -- the two validation profiles
plus unit-level checks for pure-logic capabilities:

| Profile | File | Venue |
| --- | --- | --- |
| SIL (full sim) | `profiles/sil.toml` | `sil` |
| SIL + real link | `profiles/sil-link-real.toml` | `sil-link-real` |
| Unit | (pytest) | `unit` |

PIL and HIL profiles are **DEFINED-NOT-RUN** (no hardware yet); requirements verifiable only there
are deliberately absent rather than falsely marked verified.

## Matrix

| Requirement | Statement | Method | Venue | Evidence | Status |
| --- | --- | --- | --- | --- | --- |
| REQ-COMM-HIGH-003 | Authenticated command ingress (HMAC + accepted sources) | SIL | sil | test_iss_ingress_pipeline; scenario:ingress_auth_accept | verified |
| REQ-COMM-HIGH-004 | Command ACK/NACK for every ingested command | SIL | sil | test_iss_iface_app; scenario:ingress_nack | verified |
| REQ-COMM-HIGH-002 | CCSDS framing + CRC integrity | SIL | sil | test_ccsds_codec; scenario:ingress_auth_accept | verified |
| REQ-COMM-HIGH-001 | Downlink gated by AOS visibility | SIL | sil-link-real | test_sil_closed_loop; scenario:ingress_auth_accept | verified |
| REQ-SAFE-HIGH-002 | Thermal over-limit -> SAFE + stow | SIL | sil | test_sil_closed_loop; scenario:safe_on_thermal | verified |
| REQ-AIML-GIMB-001 | Autonomous closed-loop pointing toward plume | SIL | sil | test_sil_closed_loop; scenario:closed_loop_pointing | verified |
| REQ-GIMB-HIGH-001 | ROI retention within pointing deadband | SIL | sil | test_sil_closed_loop; scenario:closed_loop_pointing | verified |
| REQ-GIMB-HIGH-003 | Runaway gimbal detection forces stow | SIL | sil | test_runaway | verified |
| REQ-COMM-CMD-001 | Command routing + ARM/EXECUTE two-step + inhibit re-check | SIL | sil | test_routing; test_sil_command_router; scenario:command_route_exec | verified |
| REQ-SAFE-EXIT-001 | Single latched SAFE; ground EXIT_SAFE gated on fault clear | SIL | sil | test_sil_command_router | verified |
| REQ-DATA-STORE-001 | Checksummed, quota'd, retention-managed product storage | unit | unit | test_storage | verified |
| REQ-DATA-LEDGER-001 | Reboot-surviving append-only fault ledger | unit | unit | test_storage | verified |
| REQ-DATA-DOWNLINK-001 | Prioritized, AOS-gated, budgeted downlink of products | SIL | sil-link-real | test_downlink; test_sil_data_system; scenario:product_downlink | verified |
| REQ-MECH-HIGH-001 | Launch-lock hazardous release + bidirectional gimbal interlock | SIL | sil | test_mechanical_app; test_sil_mechanical | verified |
| REQ-COMM-MODEL-001 | Chunked model upload -> stage -> activate -> auto-rollback | SIL | sil | test_model_deploy; test_sil_model_upload | verified |
| REQ-AIML-HIGH-004 | Model acceptance gate + load hash/contract + latency budget | unit | unit | test_model_verify; test_accept | verified |
| REQ-PLAT-QUEUE-001 | Bounded bus queues + per-type overflow policy | unit | unit | test_bus | verified |
| REQ-PLAT-SUP-001 | Thread supervision (restart->SAFE) + startup health gate | unit | unit | test_scheduler; test_health | verified |
| REQ-CONFIG-INTEGRITY-001 | Config validation: ranges, cross-field, unknown-key | unit | unit | test_config_loader | verified |
| REQ-OBS-SIL-001 | Read-only observability (bus queue-depth + state introspection) for passive SIL telemetry capture | SIL | sil | test_bus; test_analysis_recorder; test_analysis_report | verified |

## Permanent gaps

| Gap | Statement | Status |
| --- | --- | --- |
| GAP-GROUND-SEGMENT | Real ground segment is never tested; the GSE station emulator stands in for it. The `lock` axis (LaunchLock) is likewise a permanent VCRM gap -- there is no device and no config field, only this record. | gap |
