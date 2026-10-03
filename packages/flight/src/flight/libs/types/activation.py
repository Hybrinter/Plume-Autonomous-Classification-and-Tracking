"""Authority-scoped activation identity shared by payload graphs and mode messages.

Pure value records; no behavior lives here. The composition root supplies the
epoch; the external mode authority owns the sequence.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ActivationKey:
    """Authority-scoped activation identity: session epoch plus sequence.

    Attributes:
        epoch: Composition-root-provided session epoch.
        sequence: Authority-owned activation sequence within the epoch.
    """

    epoch: str
    sequence: int
