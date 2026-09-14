"""Open-loop trial harness around sim.environment.evaluate.

Not a world axis. Independent Generators per trial. No bus, mosaics, or
CaptureResult.

Contains:
  - TrialSpec, RewindThenLimbParams
  - spawn_trial_rngs
  - StepRecord, TrialRecord, shutter_pose_at, run_open_loop
"""

from sim.trial.open_loop import StepRecord, TrialRecord, run_open_loop, shutter_pose_at
from sim.trial.seeds import spawn_trial_rngs
from sim.trial.spec import RewindThenLimbParams, TrialSpec

__all__ = [
    "RewindThenLimbParams",
    "StepRecord",
    "TrialRecord",
    "TrialSpec",
    "run_open_loop",
    "shutter_pose_at",
    "spawn_trial_rngs",
]
