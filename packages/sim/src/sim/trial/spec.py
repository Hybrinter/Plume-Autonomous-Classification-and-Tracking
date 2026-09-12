"""TrialSpec: open-loop world sampling recipe. Not a world axis and not PactConfig."""

from __future__ import annotations

import math
from dataclasses import field
from typing import Literal

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass

from sim.environment.config import EnvironmentConfig

_SCHEMA = ConfigDict(extra="forbid", frozen=True)

ShutterKind = Literal["constant", "rewind_then_limb"]


@pydantic_dataclass(config=_SCHEMA)
class RewindThenLimbParams:
    """1-axis elevation slew from el_start_rad toward el_limb_rad, then hold."""

    el_start_rad: float = 0.0
    el_limb_rad: float = math.radians(45.0)
    omega_img_rad_s: float = math.radians(1.72)


@pydantic_dataclass(config=_SCHEMA)
class TrialSpec:
    """Outer Monte Carlo recipe around Environment.evaluate.

    Fields:
        n_trials: Independent world RNG streams.
        master_seed: SeedSequence entropy for spawn_trial_rngs.
        world: Named environment models.
        steps: Evaluate calls per trial.
        dt_s: Seconds advanced per step (monotonic and UTC).
        shutter: Constant elevation or rewind-then-limb.
        true_el_rad: Elevation for shutter=constant (0 at nadir).
        rewind: Params for shutter=rewind_then_limb.
        exposure_us: ShutterPose exposure.
        gain_db: ShutterPose gain.
    """

    n_trials: int
    master_seed: int
    world: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    steps: int = 1
    dt_s: float = 1.0
    shutter: ShutterKind = "constant"
    true_el_rad: float = 0.0
    rewind: RewindThenLimbParams = field(default_factory=RewindThenLimbParams)
    exposure_us: float = 13.0
    gain_db: float = 0.0
