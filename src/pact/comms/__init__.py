"""
Comms subsystem for PACT.

Provides CCSDS Space Packet encoding/decoding, a priority-ordered downlink queue with
daily byte budget enforcement, chunked model uplink with CRC verification and staged
deployment, and communication pass window scheduling.

Satisfies: REQ-COMM-HIGH-001, REQ-COMM-HIGH-002, REQ-COMM-HIGH-003, GOAL-004, GOAL-008
"""
