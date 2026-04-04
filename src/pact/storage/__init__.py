"""PACT storage subsystem.

Persists raw frames, processed tensors, and inference metadata to disk with SHA-256
checksums and an append-only JSON-lines manifest.

Satisfies: REQ-IMAG-HIGH-003, GOAL-003, GOAL-004.
"""
