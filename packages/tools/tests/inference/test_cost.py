"""Parameter and FLOP count tests."""

from tools.inference.arch.registry import build
from tools.inference.cost import count_flops, count_params


def test_unet_cost_at_32px() -> None:
    """U-Net at 32 px reports positive parameter and FLOP counts."""
    model = build("segmentor", "unet", 4).eval()
    params = count_params(model)
    flops = count_flops(model, (1, 4, 32, 32))
    assert params > 0
    assert flops > 0
