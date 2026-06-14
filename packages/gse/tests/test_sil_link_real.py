"""Blessed x86 partial: signed command uplinked over a real link, ACCEPTED ack downlinked."""

import json

from flight.libs.ccsds import decode_packet
from flight.libs.types import Ok
from gse.harness import InProcessBackend
from gse.scenario import CommandStep, Scenario, SceneSpec


def _link_real_scenario() -> Scenario:
    """A sil-link-real scenario uplinking one signed SET_THERMAL_LIMIT at frame 1."""
    return Scenario(
        name="sil-link-real-ack",
        profile="profiles/sil-link-real.toml",
        scene=SceneSpec(num_frames=8, seed=0),
        commands=(
            CommandStep(
                at_frame=1,
                command_id="SET_THERMAL_LIMIT",
                params={"limit_c": 70.0},
                source="ground",
                seq=1,
            ),
        ),
        assertions=(),
        steps=8,
        dt=1.0,
    )


def _accepted_ack_downlinked(packets: list[bytes]) -> bool:
    """True if any captured TM datagram decodes to an ACCEPTED SET_THERMAL_LIMIT ack."""
    for raw in packets:
        decoded = decode_packet(raw)
        if not isinstance(decoded, Ok):
            continue
        _header, body = decoded.value
        try:
            record = json.loads(body.decode("utf-8"))
        except ValueError, UnicodeDecodeError:
            continue
        if (
            record.get("type") == "command_ack"
            and record.get("status") == "ACCEPTED"
            and record.get("command_id") == "SET_THERMAL_LIMIT"
        ):
            return True
    return False


def test_sil_link_real_authenticated_command_acked_over_socket() -> None:
    """A signed command over a real socket link yields an ACCEPTED ack on the emulator's UDP."""
    backend = InProcessBackend()
    backend.build(_link_real_scenario(), "profiles/sil-link-real.toml")
    try:
        backend.inject_command(
            CommandStep(
                at_frame=1,
                command_id="SET_THERMAL_LIMIT",
                params={"limit_c": 70.0},
                source="ground",
                seq=1,
            )
        )
        # Event-counted, bounded poll loop (NOT a wall-clock deadline): step until the
        # emulator's UDP socket has captured the ACCEPTED downlink, up to a step budget.
        captured: list[bytes] = []
        found = False
        for i in range(20):
            backend.step(float(i + 1))
            captured.extend(backend.collect().downlink_packets)
            if _accepted_ack_downlinked(captured):
                found = True
                break
        assert found, f"no ACCEPTED SET_THERMAL_LIMIT ack in {len(captured)} captured datagrams"
    finally:
        backend.shutdown()
