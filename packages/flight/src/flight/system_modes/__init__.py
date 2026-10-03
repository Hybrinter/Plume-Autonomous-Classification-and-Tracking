"""System-mode authority: the single owner of the active system mode.

The authority receives mode requests and system-mode ground commands, decides each one with
the pure transition table, and publishes a transition record for every decision plus an
activation for every accepted transition. Peer apps reach it only over the bus.
"""
