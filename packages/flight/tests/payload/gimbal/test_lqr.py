"""Tests for the LQR gimbal control law."""

import numpy as np
from flight.libs.config import ControllerConfig, GimbalConfig
from flight.payload.gimbal import LqrController, compute_control


def test_command_clamped_to_hw_slew() -> None:
    """A large pointing error produces a command clamped to +/- hardware slew."""
    cfg = ControllerConfig()
    max_slew = GimbalConfig().max_hw_slew_rate_deg_per_s
    controller = LqrController.from_config(cfg, max_slew)
    command = np.asarray(compute_control(controller, np.array([1000.0, 1000.0, 0.0, 0.0])))
    assert command.shape == (2,)
    assert abs(float(command[0])) <= max_slew + 1e-9
    assert abs(float(command[1])) <= max_slew + 1e-9


def test_zero_error_zero_command() -> None:
    """No pointing error yields an approximately zero slew command."""
    controller = LqrController.from_config(
        ControllerConfig(), GimbalConfig().max_hw_slew_rate_deg_per_s
    )
    command = np.asarray(compute_control(controller, np.zeros(4, dtype=np.float64)))
    assert abs(float(command[0])) < 1e-9
    assert abs(float(command[1])) < 1e-9
