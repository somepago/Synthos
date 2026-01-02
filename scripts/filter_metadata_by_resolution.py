"""
Filter metadata.csv to only include images with min dimension >= threshold.

This ensures we don't train on tiny images that would need heavy upscaling.

Usage:
    python scripts/filter_metadata_by_resolution.py \
        --input data/contrastyles_full/metadata.csv \
        --output data/contrastyles_full/metadata_filtered.csv \
        --base_path data/contrastyles_full \
        --min_dim 512
"""

import argparse
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm


def get_image_dimensions(img_path):
    """Get image width and height."""
    try:
        with Image.open(img_path) as im:
            return im.size  # (width, height)
    except Exception as e:
        print(f"Error reading {img_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Filter metadata by image resolution")
    parser.add_argument("--input", type=str, required=True, help="Input metadata.csv path")
    parser.add_argument("--output", type=str, required=True, help="Output filtered metadata.csv path")
    parser.add_argument("--base_path", type=str, required=True, help="Base path for image files")
    parser.add_argument("--min_dim", type=int, default=512, help="Minimum dimension threshold (default: 512)")
    parser.add_argument("--image_column", type=str, default="video", help="Column name for image paths (default: video)")
    args = parser.parse_args()

    print(f"Loading metadata from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"  Total entries: {len(df)}")

    print(f"\nFiltering images with min dimension >= {args.min_dim}...")
    kept = []
    skipped = 0
    errors = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking images"):
        img_rel_path = row[args.image_column]
        img_path = os.path.join(args.base_path, img_rel_path)

        dims = get_image_dimensions(img_path)
        if dims is None:
            errors += 1
            continue

        w, h = dims
        min_dim = min(w, h)

        if min_dim >= args.min_dim:
            kept.append(row)
        else:
            skipped += 1

    print(f"\nResults:")
    print(f"  Kept: {len(kept)} ({100*len(kept)/len(df):.1f}%)")
    print(f"  Skipped (too small): {skipped}")
    print(f"  Errors: {errors}")

    # Create filtered dataframe
    df_filtered = pd.DataFrame(kept)
    df_filtered.to_csv(args.output, index=False)
    print(f"\nSaved filtered metadata to {args.output}")


if __name__ == "__main__":
    main()
