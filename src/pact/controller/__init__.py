"""
Controller subsystem for PACT.

Implements the gimbal safety arbiter state machine, blob tracker, EMA centroid filter,
and all safety gates that protect the gimbal from runaway or invalid commands.

Satisfies: REQ-AIML-GIMB-001 through 008, REQ-AIML-DATA-006 through 009,
           REQ-GIMB-HIGH-001 through 004
"""
