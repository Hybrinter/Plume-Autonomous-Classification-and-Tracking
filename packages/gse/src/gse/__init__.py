"""PACT ground support equipment (GSE): station emulator + deterministic scenario harness.

GSE is OUT-OF-FLIGHT test tooling. It stands in for the real ISS ground segment so the flight
software's command-ingress and downlink paths can be exercised end-to-end (sockets for PIL/HIL,
in-process for SIL). gse depends ONLY on flight.libs (CCSDS framing, the command dictionary,
build_tc_packet) and sim (scene + step_once); flight and sim must never import gse (enforced by
the flight-gse-isolation / sim-gse-isolation import-linter contracts).

Satisfies: REQ-VAL-GSE-001.
"""
