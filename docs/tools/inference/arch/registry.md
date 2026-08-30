# tools.inference.arch.registry

**Source:** `packages/tools/src/tools/inference/arch/registry.py`
**Kind:** module

## Purpose

This module maps a train kind and architecture name to a network builder.
Architecture names form a grammar. `known()` returns a curated catalog.
`resolve_arch` accepts any name that parses.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `DEFAULT_ARCH` | constant | Kind to default architecture name |
| `DEFAULT_BASE_WIDTH` | constant | Default scratch U-Net stem width (64) |
| `DEFAULT_DEPTH` | constant | Default scratch U-Net stage count (4) |
| `DEFAULT_DECODER_WIDTH` | constant | Default encoder U-Net decoder width (16) |
| `UNetSpec` | class | Parsed scratch U-Net modifiers |
| `EncoderUNetSpec` | class | Parsed ResNet-encoder U-Net modifiers |
| `DilatedSpec` | class | Parsed dilated segmentor modifiers |
| `parse_classifier` | function | Parse a classifier grammar name |
| `parse_segmentor` | function | Parse a segmentor grammar name |
| `default_arch` | function | Default name for a kind |
| `known` | function | Curated catalog of kind/name pairs |
| `resolve_arch` | function | Fill an empty name and validate grammar |
| `build` | function | Construct an untrained logits graph |

## Inputs and outputs

`parse_classifier(name) -> BackboneSpec | CompactSpec`. Raises `ValueError` on
an unknown family or modifier.

`parse_segmentor(name) -> UNetSpec | EncoderUNetSpec | DilatedSpec`. Raises
`ValueError` on an unknown family or modifier.

`default_arch(kind) -> str`.

`known() -> frozenset[tuple[str, str]]`.

`resolve_arch(kind, arch) -> str`.

`build(kind, arch, in_channels) -> nn.Module`. Raises `ValueError` on an unknown
kind or unparsable name.

## Behavior

1. Empty `arch` selects `resnet50` for the classifier and `unet` for the
   segmentor.
2. **Torchvision classifier family.** A backbone name with an optional `_pt`
   suffix for ImageNet weights. Backbones: `resnet18`, `resnet34`, `resnet50`,
   `mobilenetv3_small`, `mobilenetv3_large`, `efficientnet_b0`,
   `shufflenetv2_x0_5`. Examples: `resnet18`, `mobilenetv3_small_pt`.
3. **Compact classifier family.** Name `pactnet` with underscore-separated
   modifiers: `w<N>` stem width (default 16), `d<N>` stage count (default 4),
   `full` for dense convolutions. Examples: `pactnet`, `pactnet_w32_d5_full`.
4. **Scratch U-Net family.** Name `unet` with underscore-separated modifiers:
   `w<N>` stem width (default 64), `d<N>` stage count (default 4), `sep` for
   depthwise-separable convolutions. Examples: `unet`, `unet_w16_d3_sep`.
5. **Encoder U-Net family.** Names `runet18`, `runet34`, `runet50` with
   modifiers: `pt` for ImageNet encoder weights, `x<N>` decoder width (default
   16), `sep` for separable decoder convolutions. Examples: `runet18_pt_x32`.
6. **Dilated segmentor family.** Name `dilatenet` with underscore-separated
   modifiers: `w<N>` stem width (default 32), `d<N>` dilated-block count
   (default 4), `s<N>` output stride of 4 or 8 (default 4), `full` for dense
   convolutions. Examples: `dilatenet`, `dilatenet_w32_d6_s8`.
7. `build` dispatches to `build_backbone`, `build_compact_classifier`,
   `build_segmentor`, `build_encoder_segmentor`, or `build_dilated_segmentor`
   from the parsed spec.

## Errors and faults

`ValueError` on an unknown kind, backbone, segmentor family, or modifier token.

## Messages

None.

## Configuration

`DEFAULT_ARCH` maps `classifier` to `resnet50` and `segmentor` to `unet`.
`known()` lists representative points across both classifier families and all
three segmentor families. Names outside the catalog remain valid when they parse.

## Constraints

Adding a backbone is one torchvision constructor plus a registry row in the
catalog. Adding a classifier or segmentor family needs a parser branch and a
builder call.

## Related documents

- [`tools.inference.arch`](../arch.md)
- [`tools.inference.arch.classifier`](classifier.md)
- [`tools.inference.arch.compact`](compact.md)
- [`tools.inference.arch.unet`](unet.md)
- [`tools.inference.arch.encoder_unet`](encoder_unet.md)
- [`tools.inference.arch.dilated`](dilated.md)
- [`tools.inference.train`](../train.md)
- [`tools.inference.export`](../export.md)
