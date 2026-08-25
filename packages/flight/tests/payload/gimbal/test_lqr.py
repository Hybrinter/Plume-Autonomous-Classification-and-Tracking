"""Tests for the LQR gimbal control law."""

import numpy as np
from flight.libs.config import ControllerConfig
from flight.payload.gimbal import LqrController, compute_control


def test_command_clamped_to_max_slew() -> None:
    """A large pointing error produces a command clamped to +/- max_slew_deg_s."""
    cfg = ControllerConfig()
    controller = LqrController.from_config(cfg)
    command = np.asarray(compute_control(controller, np.array([1000.0, 1000.0, 0.0, 0.0])))
    assert command.shape == (2,)
    assert abs(float(command[0])) <= cfg.max_slew_deg_s + 1e-9
    assert abs(float(command[1])) <= cfg.max_slew_deg_s + 1e-9


def test_zero_error_zero_command() -> None:
    """No pointing error yields an approximately zero slew command."""
    controller = LqrController.from_config(ControllerConfig())
    command = np.asarray(compute_control(controller, np.zeros(4, dtype=np.float64)))
    assert abs(float(command[0])) < 1e-9
    assert abs(float(command[1])) < 1e-9
