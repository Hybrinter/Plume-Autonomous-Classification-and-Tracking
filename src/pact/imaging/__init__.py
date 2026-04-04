"""
Imaging subsystem for PACT.

Provides camera abstraction, raw frame capture loop, and the imaging process entry point.
Hardware-dependent code (FlirBlackflyCamera) is isolated behind the AbstractCamera Protocol
so that all other code (including tests) uses MockCamera without requiring PySpin.

Satisfies: REQ-AIML-IMAG-001, REQ-AIML-IMAG-002
"""
