"""Result[T, E] / Ok[T] / Err[E] explicit success-or-error wrapper types.

Library code returns these types instead of
raising, so callers can pattern-match on success vs failure.

Satisfies: REQ-AIML-COMP-001, REQ-AIML-COMP-002 (type-safety foundation for all subsystems).

No other flight module is imported here. This module is a dependency root.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass
from typing import Generic, TypeVar, Union

# ---------------------------------------------------------------------------
# Generic type variables for Result
# ---------------------------------------------------------------------------

T = TypeVar("T")
E = TypeVar("E")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ok(Generic[T]):  # noqa: UP046  (explicit Generic form is the stable public contract)
    """Successful result wrapper holding the success value."""

    value: T


@dataclass(frozen=True)
class Err(Generic[E]):  # noqa: UP046  (explicit Generic form is the stable public contract)
    """Error result wrapper holding the error value."""

    error: E


# Result is a type alias; cannot be parameterised at runtime but is valid for type checkers.
Result = Union[Ok[T], Err[E]]  # noqa: UP007  (explicit Union form is the stable public contract)
