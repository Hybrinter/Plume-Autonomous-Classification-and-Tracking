"""Download the HSG-AIML plume segmentation dataset from Zenodo.

Downloads GeoTIFF imagery and polygon annotations for the HSG-AIML dataset
(Zenodo DOI: 10.5281/zenodo.4250706) to the specified output directory.
Verifies the MD5 checksum of the downloaded archive before extraction.

Usage
-----
    python scripts/download_dataset.py [--output-dir data/raw]

Satisfies: §7 of PACT_SW_ARCH.md (scripts/download_dataset.py)
"""

from __future__ import annotations

# stdlib
import argparse
import sys


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download HSG-AIML dataset from Zenodo to the output directory."
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Target directory for the downloaded dataset (default: data/raw)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: download and verify the HSG-AIML dataset."""
    args = parse_args()
    print(f"Downloading HSG-AIML dataset from Zenodo (DOI: 10.5281/zenodo.4250706)...")
    print(f"Target directory: {args.output_dir}")
    # TODO: implement
    # 1. Use requests.get() to fetch the Zenodo API endpoint for record 4250706
    # 2. Parse the JSON response to find the download URL for the main archive
    # 3. Stream-download the archive with tqdm progress bar
    # 4. Verify MD5 checksum of the downloaded archive
    # 5. Extract to args.output_dir using zipfile or tarfile
    # 6. Print "Download complete. N files extracted to <output_dir>."
    print("TODO: download not yet implemented. See dataset.download_dataset() stub.")
    sys.exit(0)


if __name__ == "__main__":
    main()
