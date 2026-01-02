"""
Download full ContraStyles dataset (500k images) using img2dataset.

This downloads all images locally so you don't need streaming.
Estimated size: ~150-300GB depending on image sizes.
"""

import os
import subprocess
from datasets import load_dataset
import pandas as pd

# Output paths
OUTPUT_DIR = "data/contrastyles_full"
PARQUET_PATH = f"{OUTPUT_DIR}/urls.parquet"
IMAGES_DIR = f"{OUTPUT_DIR}/images"
METADATA_PATH = f"{OUTPUT_DIR}/metadata.csv"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    print("Step 1: Loading dataset metadata from HuggingFace...")
    ds = load_dataset("tomg-group-umd/ContraStyles", split="train")
    print(f"  Found {len(ds)} samples")

    print("\nStep 2: Creating URL parquet for img2dataset...")
    # img2dataset expects: url, caption columns
    df = pd.DataFrame({
        "url": ds["url"],
        "caption": ds["caption"],
        "key": ds["key"],
    })
    df.to_parquet(PARQUET_PATH)
    print(f"  Saved to {PARQUET_PATH}")

    print("\nStep 3: Downloading images with img2dataset...")
    print("  This will take a while for 500k images...")

    # img2dataset command - no resizing, keep original images, resumable
    cmd = [
        "img2dataset",
        "--url_list", PARQUET_PATH,
        "--output_folder", IMAGES_DIR,
        "--output_format", "files",  # Save as individual files
        "--input_format", "parquet",
        "--url_col", "url",
        "--caption_col", "caption",
        "--disable_all_reencoding", "True",  # Keep original format, no re-encoding
        "--thread_count", "64",  # Parallel downloads
        "--retries", "3",  # Retry failed downloads
        "--incremental_mode", "incremental",  # Skip already downloaded shards (resume support)
        "--timeout", "30",  # Longer timeout for slow servers
    ]

    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    print("\nStep 4: Generating metadata.csv for training...")
    generate_metadata()

    print(f"\nDone! Dataset saved to {OUTPUT_DIR}")
    print(f"  - Images: {IMAGES_DIR}")
    print(f"  - Metadata: {METADATA_PATH}")

def generate_metadata():
    """Generate metadata.csv from downloaded images."""
    import glob

    # Find all downloaded images
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        image_files.extend(glob.glob(f"{IMAGES_DIR}/**/{ext}", recursive=True))

    print(f"  Found {len(image_files)} downloaded images")

    # Load original captions
    ds = load_dataset("tomg-group-umd/ContraStyles", split="train")
    key_to_caption = {row["key"]: row["caption"] for row in ds}

    # Match images to captions
    records = []
    for img_path in image_files:
        # Extract key from filename (img2dataset uses the key)
        basename = os.path.basename(img_path)
        key = os.path.splitext(basename)[0]

        # Get relative path for training
        rel_path = os.path.relpath(img_path, OUTPUT_DIR)

        caption = key_to_caption.get(key, "")
        if caption:
            records.append({
                "video": rel_path,  # Using "video" key for compatibility with training
                "prompt": caption,
            })

    df = pd.DataFrame(records)
    df.to_csv(METADATA_PATH, index=False)
    print(f"  Saved {len(df)} entries to {METADATA_PATH}")

if __name__ == "__main__":
    main()
