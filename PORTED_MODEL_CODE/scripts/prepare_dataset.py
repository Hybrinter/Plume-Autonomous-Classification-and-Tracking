#!/usr/bin/env python3
"""
Prepare Zenodo 4250706 (or compatible) smoke plume data for training.

Expected source layout (after extracting images + segmentation_labels):
  SOURCE/images/positive/*.tif
  SOURCE/images/negative/*.tif
  SOURCE/segmentation_labels/*.json

Produces:
  OUTPUT/classification/{train,val,test}/{positive,negative}/*.tif
  OUTPUT/segmentation/{train,val,test}/images/{positive,negative}/*.tif
  OUTPUT/segmentation/{train,val,test}/labels/*.json

Split rule (same as README): group by site id (prefix before first '_' in the
GeoTIFF basename) so all time steps for one site stay in one split.

JSON label URLs use ':' in times; on-disk GeoTIFFs use '_'. The training code
normalizes keys accordingly.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict


def site_id_from_stem(stem: str) -> str:
    return stem.split("_", 1)[0]


def json_url_to_tif_basename(url: str) -> str:
    key = "-".join(url.split("-")[1:]).replace(".png", ".tif")
    return key.replace(":", "_")


def link_or_copy(src: str, dst: str, mode: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)
        return
    if mode == "hardlink":
        if os.path.exists(dst):
            return
        os.link(src, dst)
        return
    raise ValueError(mode)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        default=os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "4250706")),
        help="Zenodo root containing images/ and segmentation_labels/",
    )
    p.add_argument(
        "--output",
        default=os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data")),
        help="Output root for prepared splits",
    )
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", choices=("copy", "symlink", "hardlink"), default="copy")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--force",
        action="store_true",
        help="Remove existing --output directory before writing",
    )
    args = p.parse_args()

    tr, vr, te = args.train_ratio, args.val_ratio, args.test_ratio
    if abs(tr + vr + te - 1.0) > 1e-6:
        print("train + val + test ratios must sum to 1.0", file=sys.stderr)
        return 1

    src_pos = os.path.join(args.source, "images", "positive")
    src_neg = os.path.join(args.source, "images", "negative")
    src_lbl = os.path.join(args.source, "segmentation_labels")

    for path, label in [
        (src_pos, "positive images"),
        (src_neg, "negative images"),
        (src_lbl, "segmentation_labels"),
    ]:
        if not os.path.isdir(path):
            print(f"Missing {label} directory: {path}", file=sys.stderr)
            return 1

    # site -> list of (class_dir_name, filename, full_src_path)
    site_files: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for cls, folder in (("positive", src_pos), ("negative", src_neg)):
        for fn in os.listdir(folder):
            if not fn.lower().endswith(".tif"):
                continue
            stem = os.path.splitext(fn)[0]
            sid = site_id_from_stem(stem)
            site_files[sid].append((cls, fn, os.path.join(folder, fn)))

    # site -> json filenames that belong to that site (from basename)
    site_labels: dict[str, list[str]] = defaultdict(list)
    json_to_tif: dict[str, str] = {}
    for fn in os.listdir(src_lbl):
        if not fn.endswith(".json"):
            continue
        jpath = os.path.join(src_lbl, fn)
        with open(jpath, encoding="utf-8") as f:
            data = json.load(f)
        url = data["data"]["image"]
        tif_base = json_url_to_tif_basename(url)
        json_to_tif[fn] = tif_base
        sid = site_id_from_stem(os.path.splitext(tif_base)[0])
        site_labels[sid].append(fn)

    site_ids = sorted(site_files.keys())
    rng = random.Random(args.seed)
    rng.shuffle(site_ids)

    n = len(site_ids)
    n_train = int(round(tr * n))
    n_val = int(round(vr * n))
    train_sites = set(site_ids[:n_train])
    val_sites = set(site_ids[n_train : n_train + n_val])
    test_sites = set(site_ids[n_train + n_val :])

    split_for_site = {}
    for s in train_sites:
        split_for_site[s] = "train"
    for s in val_sites:
        split_for_site[s] = "val"
    for s in test_sites:
        split_for_site[s] = "test"

    def out_class_dir(split: str, cls: str) -> str:
        return os.path.join(args.output, "classification", split, cls)

    def out_seg_img(split: str, cls: str) -> str:
        return os.path.join(args.output, "segmentation", split, "images", cls)

    def out_seg_lbl(split: str) -> str:
        return os.path.join(args.output, "segmentation", split, "labels")

    planned: list[tuple[str, str, str]] = []
    for sid, entries in site_files.items():
        sp = split_for_site[sid]
        for cls, fn, src in entries:
            planned.append(
                (src, os.path.join(out_class_dir(sp, cls), fn), f"classification/{sp}/{cls}")
            )
            planned.append(
                (src, os.path.join(out_seg_img(sp, cls), fn), f"segmentation/{sp}/images/{cls}")
            )

    for sid, jfns in site_labels.items():
        sp = split_for_site[sid]
        for jfn in jfns:
            src = os.path.join(src_lbl, jfn)
            planned.append((src, os.path.join(out_seg_lbl(sp), jfn), f"segmentation/{sp}/labels"))

    if args.dry_run:
        print(f"Sites: {n}  train={len(train_sites)} val={len(val_sites)} test={len(test_sites)}")
        print(f"File operations (incl. duplicates for cls+seg image copies): {len(planned)}")
        train_lbl = os.path.join(args.output, "segmentation", "train", "labels")
        lbl_train = sum(1 for _, d, _ in planned if d.startswith(train_lbl + os.sep))
        print(f"JSON labels in train split: {lbl_train}")
        return 0

    if os.path.exists(args.output):
        if not args.force:
            print(
                f"Output exists: {args.output}\nUse --force to replace, or pick a new --output.",
                file=sys.stderr,
            )
            return 1
        shutil.rmtree(args.output)

    for src, dst, _role in planned:
        link_or_copy(src, dst, args.mode)

    # Sanity: every JSON should point to a file that exists in the same split
    missing = []
    for jfn, tif_base in json_to_tif.items():
        sid = site_id_from_stem(os.path.splitext(tif_base)[0])
        sp = split_for_site[sid]
        tpath = os.path.join(out_seg_img(sp, "positive"), tif_base)
        if not os.path.isfile(tpath):
            missing.append((jfn, tpath))
    if missing:
        print("Warning: label JSON without matching positive TIF in split:", file=sys.stderr)
        for jfn, tpath in missing[:10]:
            print(f"  {jfn} -> {tpath}", file=sys.stderr)
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more", file=sys.stderr)

    print(f"Done. Output: {args.output}")
    print("Use smoke_detection.common.paths.DATASET_ROOT from training code.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
