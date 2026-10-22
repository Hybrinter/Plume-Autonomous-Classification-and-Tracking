"""Dual-branch VIS+SWIR variants of the conditioned flight families.

The paired VIS+SWIR build stores one ``(4, H, W)`` tensor per row on the VIS
grid: BLUE, GREEN, RED, then SWIR. The single-stem families mix all four
planes in their first convolution (early fusion). The dual variants split the
tensor instead. A VIS branch reads planes 0-2 and a SWIR branch reads plane 3.
Each branch runs its own stem and one strided block. The branch features are
concatenated at output stride 4 and mixed by a 1x1 convolution, and the rest of
the network is the usual ``pactnet`` or ``dilatenet`` body (mid fusion).

Fusing at stride 4 rather than deeper is set by the training tiles: paired rows
are 18 x 27 to 39 x 40 VIS samples, and a stride-16 fusion point would leave
one or two cells.

Two training-only perturbations model the flight pairing errors. Both are
no-ops in eval mode, so analysis and export see a deterministic graph:

  - Registration jitter shifts the SWIR plane of each sample by an integer
    offset in ``[-jitter_px, jitter_px]`` per axis, with edge replication. The
    default of 1 VIS sample is about the frame-timing offset expected between
    unsynchronized 32 Hz and 55 Hz frames at the coarse bins.
  - Modality dropout zeroes the VIS branch features of a sample with
    probability ``modality_dropout`` and the SWIR branch features with the same
    probability, never both, so the body learns to run on one camera.

The names are modifiers on the existing families: ``pactnet_dual`` and
``dilatenet_dual``. They accept the family's other modifiers.

Contains:
  - DUAL_INPUT_CHANNELS: the band count a dual graph requires.
  - DEFAULT_JITTER_PX / DEFAULT_MODALITY_DROPOUT: training perturbation defaults.
  - shift_planes: per-sample integer shift with edge replication.
  - DualStem: two branch stems fused at stride 4.
  - DualPactNet: dual-branch ``pactnet`` classifier.
  - DualDilatedSegmentor: dual-branch ``dilatenet`` segmentor.
  - build_dual_classifier / build_dual_segmentor: construct one from a spec.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import torch
from torch import nn

from tools.ml_models.arch.blocks import conv_norm_relu
from tools.ml_models.arch.compact import CompactSpec, compact_stage_widths
from tools.ml_models.arch.dilated import DilatedSpec, dilation_rates
from tools.ml_models.arch.film import GsdFilm, resolve_gsd

DUAL_INPUT_CHANNELS = 4

DEFAULT_JITTER_PX = 1
DEFAULT_MODALITY_DROPOUT = 0.1

_VIS_PLANES = 3
_HEAD_DROPOUT = 0.2
_BODY_MULTIPLIER = 2


def _block(in_channels: int, out_channels: int, stride: int, separable: bool) -> nn.Module:
    """Return one batch-normalised, rectified 3x3 stage."""
    return conv_norm_relu(
        in_channels, out_channels, stride=stride, separable=separable, mid_norm=separable
    )


def _check_channels(in_channels: int) -> None:
    """Raise unless the input is the four-plane BLUE/GREEN/RED/SWIR stack."""
    if in_channels != DUAL_INPUT_CHANNELS:
        raise ValueError(
            f"dual-branch graphs need {DUAL_INPUT_CHANNELS} input planes "
            f"(BLUE, GREEN, RED, SWIR); got {in_channels}"
        )


def shift_planes(planes: torch.Tensor, offsets: torch.Tensor, max_px: int) -> torch.Tensor:
    """Shift each sample by an integer offset with edge replication.

    Args:
        planes: Input of shape ``(N, C, H, W)``.
        offsets: Integer offsets of shape ``(N, 2)`` as ``(dy, dx)``, each in
            ``[-max_px, max_px]``.
        max_px: Largest absolute offset.

    Returns:
        torch.Tensor: Shifted planes, same shape as ``planes``. Output pixel
        ``(y, x)`` reads input pixel ``(y - dy, x - dx)``, clamped to the edge.
    """
    if max_px < 1:
        return planes
    height, width = planes.shape[-2:]
    padded = nn.functional.pad(planes, (max_px, max_px, max_px, max_px), mode="replicate")
    starts = (max_px - offsets).tolist()
    return torch.stack(
        [
            padded[index, :, y0 : y0 + height, x0 : x0 + width]
            for index, (y0, x0) in enumerate(starts)
        ]
    )


class DualStem(nn.Module):
    """VIS and SWIR branch stems fused at output stride 4.

    Each branch is a dense stride-2 stem, a GSD FiLM, and one stride-2 block.
    The concatenated branch features pass through a 1x1 convolution, batch-norm,
    and ReLU to ``out_width`` channels.
    """

    def __init__(
        self,
        vis_width: int,
        swir_width: int,
        out_width: int,
        separable: bool,
        jitter_px: int = DEFAULT_JITTER_PX,
        modality_dropout: float = DEFAULT_MODALITY_DROPOUT,
    ) -> None:
        """Build both branches and the fusion layer.

        Args:
            vis_width: VIS stem width; the VIS branch ends at twice this.
            swir_width: SWIR stem width; the SWIR branch ends at twice this.
            out_width: Fused feature width.
            separable: Depthwise-separable strided blocks.
            jitter_px: Largest training-time SWIR shift in VIS samples.
            modality_dropout: Per-branch training-time drop probability.

        Raises:
            ValueError: If a width is below one, ``jitter_px`` is negative, or
                ``modality_dropout`` is outside ``[0, 0.5]``.
        """
        super().__init__()
        if min(vis_width, swir_width, out_width) < 1:
            raise ValueError("dual stem widths must be at least 1")
        if jitter_px < 0:
            raise ValueError(f"jitter_px must be nonnegative, got {jitter_px}")
        if not 0.0 <= modality_dropout <= 0.5:
            raise ValueError(f"modality_dropout must lie in [0, 0.5], got {modality_dropout}")
        self.jitter_px = jitter_px
        self.modality_dropout = modality_dropout
        self.vis_stem = _block(_VIS_PLANES, vis_width, 2, separable=False)
        self.vis_film = GsdFilm(vis_width)
        self.vis_down = _block(vis_width, 2 * vis_width, 2, separable=separable)
        self.swir_stem = _block(1, swir_width, 2, separable=False)
        self.swir_film = GsdFilm(swir_width)
        self.swir_down = _block(swir_width, 2 * swir_width, 2, separable=separable)
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * (vis_width + swir_width), out_width, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_width),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, gsd: torch.Tensor) -> torch.Tensor:
        """Map the four-plane stack to fused stride-4 features.

        Args:
            x: Input of shape ``(N, 4, H, W)``.
            gsd: Encoded VIS GSD of shape ``(N, 2)``. Both branches use it; the
                SWIR to VIS GSD ratio is fixed by the optics.

        Returns:
            torch.Tensor: Features of shape ``(N, out_width, ceil(H/4), ceil(W/4))``.
        """
        vis_in = x[:, :_VIS_PLANES]
        swir_in = x[:, _VIS_PLANES:]
        if self.training and self.jitter_px > 0:
            offsets = torch.randint(
                -self.jitter_px, self.jitter_px + 1, (x.shape[0], 2), device=x.device
            )
            swir_in = shift_planes(swir_in, offsets, self.jitter_px)
        vis = self.vis_down(self.vis_film(self.vis_stem(vis_in), gsd))
        swir = self.swir_down(self.swir_film(self.swir_stem(swir_in), gsd))
        if self.training and self.modality_dropout > 0.0:
            draw = torch.rand(x.shape[0], 1, 1, 1, device=x.device)
            vis = vis * (draw >= self.modality_dropout).to(vis.dtype)
            keep_swir = (draw < self.modality_dropout) | (draw >= 2 * self.modality_dropout)
            swir = swir * keep_swir.to(swir.dtype)
        fused: torch.Tensor = self.fuse(torch.cat([vis, swir], dim=1))
        return fused


class DualPactNet(nn.Module):
    """Dual-branch ``pactnet``: a :class:`DualStem` and the compact body.

    The stem replaces the ``pactnet`` stem and first strided stage. The rest
    matches :class:`~tools.ml_models.arch.compact.PactNet`: one stride-1 block
    at the fused width, then a strided and a stride-1 block per remaining
    stage, a head FiLM, feature-map dropout, a 1x1 logit head, and a max over
    cells.
    """

    def __init__(
        self,
        base_width: int,
        depth: int,
        separable: bool,
        jitter_px: int = DEFAULT_JITTER_PX,
        modality_dropout: float = DEFAULT_MODALITY_DROPOUT,
    ) -> None:
        """Build the dual stem and the compact body.

        Args:
            base_width: VIS stem width; the SWIR stem is half, at least 1.
            depth: Strided stage count including the stem, at least 2.
            separable: Depthwise-separable body convolutions.
            jitter_px: Largest training-time SWIR shift in VIS samples.
            modality_dropout: Per-branch training-time drop probability.

        Raises:
            ValueError: If ``depth`` is below two or a width is below one.
        """
        super().__init__()
        if depth < 2:
            raise ValueError(f"dual pactnet depth must be at least 2, got {depth}")
        widths = compact_stage_widths(base_width, depth)
        self.stem = DualStem(
            base_width,
            max(1, base_width // 2),
            widths[1],
            separable,
            jitter_px=jitter_px,
            modality_dropout=modality_dropout,
        )
        stages: list[nn.Module] = [_block(widths[1], widths[1], 1, separable)]
        for index in range(2, depth):
            stages.append(_block(widths[index - 1], widths[index], 2, separable))
            stages.append(_block(widths[index], widths[index], 1, separable))
        self.body = nn.Sequential(*stages)
        self.head_film = GsdFilm(widths[-1])
        self.dropout = nn.Dropout(_HEAD_DROPOUT)
        self.head = nn.Conv2d(widths[-1], 1, kernel_size=1)

    def spatial(self, x: torch.Tensor, gsd: torch.Tensor | None = None) -> torch.Tensor:
        """Map the four-plane stack to one logit per strided cell.

        Args:
            x: Input of shape ``(N, 4, H, W)``.
            gsd: Encoded GSD of shape ``(N, 2)``; None uses the reference.

        Returns:
            torch.Tensor: Logits of shape ``(N, 1, h, w)``.
        """
        gsd = resolve_gsd(x, gsd)
        features = self.body(self.stem(x, gsd))
        logits: torch.Tensor = self.head(self.dropout(self.head_film(features, gsd)))
        return logits

    def forward(self, x: torch.Tensor, gsd: torch.Tensor | None = None) -> torch.Tensor:
        """Map the four-plane stack to one logit per sample.

        Args:
            x: Input of shape ``(N, 4, H, W)``.
            gsd: Encoded GSD of shape ``(N, 2)``; None uses the reference.

        Returns:
            torch.Tensor: Logits of shape ``(N, 1)``, the maximum of
            :meth:`spatial` over cells.
        """
        logits: torch.Tensor = self.spatial(x, gsd).amax(dim=(2, 3))
        return logits


class DualDilatedSegmentor(nn.Module):
    """Dual-branch ``dilatenet``: a :class:`DualStem` and the dilated body.

    The stem replaces the ``dilatenet`` stem and first strided block and emits
    the body width. Output stride 8 adds one strided block. The dilated blocks,
    head FiLM, 1x1 head, and bilinear resize to the input size match
    :class:`~tools.ml_models.arch.dilated.DilatedSegmentor`. The logit plane is
    on the VIS grid.
    """

    def __init__(
        self,
        out_channels: int,
        base_width: int,
        blocks: int,
        output_stride: int,
        separable: bool,
        jitter_px: int = DEFAULT_JITTER_PX,
        modality_dropout: float = DEFAULT_MODALITY_DROPOUT,
    ) -> None:
        """Build the dual stem, the dilated body, and the logit head.

        Args:
            out_channels: Output plane count.
            base_width: VIS stem width; the SWIR stem is half, at least 1.
            blocks: Dilated block count.
            output_stride: 4 or 8.
            separable: Depthwise-separable body convolutions.
            jitter_px: Largest training-time SWIR shift in VIS samples.
            modality_dropout: Per-branch training-time drop probability.

        Raises:
            ValueError: If ``output_stride`` is not 4 or 8, or a width or
                block count is below one.
        """
        super().__init__()
        if output_stride not in (4, 8):
            raise ValueError(f"output_stride must be 4 or 8, got {output_stride}")
        body_width = base_width * _BODY_MULTIPLIER
        self.stem = DualStem(
            base_width,
            max(1, base_width // 2),
            body_width,
            separable,
            jitter_px=jitter_px,
            modality_dropout=modality_dropout,
        )
        stages: list[nn.Module] = []
        if output_stride == 8:
            stages.append(_block(body_width, body_width, 2, separable))
        for rate in dilation_rates(blocks):
            stages.append(
                conv_norm_relu(
                    body_width,
                    body_width,
                    stride=1,
                    dilation=rate,
                    separable=separable,
                    mid_norm=separable,
                )
            )
        self.body = nn.Sequential(*stages)
        self.head_film = GsdFilm(body_width)
        self.head = nn.Conv2d(body_width, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, gsd: torch.Tensor | None = None) -> torch.Tensor:
        """Map the four-plane stack to a full-resolution logit plane.

        Args:
            x: Input of shape ``(N, 4, H, W)``.
            gsd: Encoded GSD of shape ``(N, 2)``; None uses the reference.

        Returns:
            torch.Tensor: Logits of shape ``(N, out_channels, H, W)``.
        """
        gsd = resolve_gsd(x, gsd)
        size = x.shape[-2:]
        features = self.body(self.stem(x, gsd))
        logits = self.head(self.head_film(features, gsd))
        resized: torch.Tensor = nn.functional.interpolate(
            logits, size=size, mode="bilinear", align_corners=False
        )
        return resized


def build_dual_classifier(spec: CompactSpec, in_channels: int) -> nn.Module:
    """Return an untrained :class:`DualPactNet` for a parsed ``pactnet_dual`` spec.

    Args:
        spec: Parsed compact specification.
        in_channels: Input band count; must be 4.

    Returns:
        nn.Module: Network mapping ``(N, 4, H, W)`` to ``(N, 1)`` logits.

    Raises:
        ValueError: If ``in_channels`` is not 4 or the spec depth is below 2.
    """
    _check_channels(in_channels)
    return DualPactNet(spec.base_width, spec.depth, spec.separable)


def build_dual_segmentor(spec: DilatedSpec, in_channels: int, out_channels: int = 1) -> nn.Module:
    """Return an untrained :class:`DualDilatedSegmentor` for a ``dilatenet_dual`` spec.

    Args:
        spec: Parsed dilated specification.
        in_channels: Input band count; must be 4.
        out_channels: Output plane count.

    Returns:
        nn.Module: Network mapping ``(N, 4, H, W)`` to ``(N, out_channels, H, W)``.

    Raises:
        ValueError: If ``in_channels`` is not 4.
    """
    _check_channels(in_channels)
    return DualDilatedSegmentor(
        out_channels, spec.base_width, spec.blocks, spec.output_stride, spec.separable
    )
