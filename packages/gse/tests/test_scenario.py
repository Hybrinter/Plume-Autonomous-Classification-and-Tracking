"""load_scenario parses a scenario TOML into the typed dataclasses with assertion tags."""

from pathlib import Path

from gse.scenario import Assertion, CommandStep, Scenario, SceneSpec, load_scenario

_SAMPLE = """\
name = "thermal_safe"
profile = "sil"
steps = 6
dt = 1.0

[scene]
num_frames = 6
seed = 0

[[commands]]
at_frame = 1
command_id = "SET_THERMAL_LIMIT"
source = "ground"
seq = 1
params = { limit_c = 70.0 }

[[assertions]]
id = "mode_goes_safe"
kind = "mode_is"
value = "SAFE"
tag = "frame-portable"

[[assertions]]
id = "ack_is_fast"
kind = "ack_within_seconds"
value = 2.0
tag = "realtime-only"
"""


def test_load_scenario_parses_all_fields(tmp_path: Path) -> None:
    """A sample scenario TOML round-trips into Scenario with both assertion tags preserved."""
    path = tmp_path / "thermal_safe.toml"
    path.write_text(_SAMPLE, encoding="ascii")

    scenario = load_scenario(str(path))

    assert isinstance(scenario, Scenario)
    assert scenario.name == "thermal_safe"
    assert scenario.profile == "sil"
    assert scenario.steps == 6
    assert scenario.dt == 1.0

    assert scenario.scene == SceneSpec(num_frames=6, seed=0)

    assert scenario.commands == (
        CommandStep(
            at_frame=1,
            command_id="SET_THERMAL_LIMIT",
            params={"limit_c": 70.0},
            source="ground",
            seq=1,
        ),
    )

    assert len(scenario.assertions) == 2
    assert scenario.assertions[0] == Assertion(
        id="mode_goes_safe", kind="mode_is", value="SAFE", tag="frame-portable"
    )
    assert scenario.assertions[1].tag == "realtime-only"
    assert scenario.assertions[1].kind == "ack_within_seconds"


def test_scene_readings_default_to_nominal_singletons() -> None:
    """A SceneSpec built without readings defaults to (20.0,) thermal and (10.0,) power."""
    scene = SceneSpec(num_frames=3, seed=0)
    assert scene.thermal_readings == (20.0,)
    assert scene.power_readings == (10.0,)


def test_load_scenario_parses_scene_readings(tmp_path: Path) -> None:
    """thermal_readings/power_readings under [scene] parse into float tuples on the SceneSpec."""
    sample = (
        'name = "hot"\n'
        'profile = "sil"\n'
        "steps = 3\n"
        "dt = 1.0\n"
        "[scene]\n"
        "num_frames = 3\n"
        "seed = 0\n"
        "thermal_readings = [95.0, 96.0]\n"
        "power_readings = [12.0]\n"
    )
    path = tmp_path / "hot.toml"
    path.write_text(sample, encoding="ascii")

    scenario = load_scenario(str(path))

    assert scenario.scene.thermal_readings == (95.0, 96.0)
    assert scenario.scene.power_readings == (12.0,)


def test_load_scenario_defaults_readings_when_absent(tmp_path: Path) -> None:
    """A [scene] table without readings yields the SceneSpec nominal-singleton defaults."""
    path = tmp_path / "thermal_safe.toml"
    path.write_text(_SAMPLE, encoding="ascii")

    scenario = load_scenario(str(path))

    assert scenario.scene.thermal_readings == (20.0,)
    assert scenario.scene.power_readings == (10.0,)
