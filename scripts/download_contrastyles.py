#!/usr/bin/env python3
"""
Download a subset of ContraStyles dataset for I2I training.
Images are stored as URLs, so we need to download them.
"""

import os
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from datasets import load_dataset
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_image(url, save_path, timeout=10):
    """Download image from URL and save to path."""
    try:
        response = requests.get(url, timeout=timeout, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P', 'LA'):
                img = img.convert('RGB')
            img.save(save_path, 'JPEG', quality=95)
            return True
    except Exception as e:
        pass
    return False


def main():
    parser = argparse.ArgumentParser(description="Download ContraStyles subset")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to download")
    parser.add_argument("--output_dir", type=str, default="data/contrastyles", help="Output directory")
    parser.add_argument("--workers", type=int, default=8, help="Number of download workers")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print(f"Loading ContraStyles dataset...")
    dataset = load_dataset("tomg-group-umd/ContraStyles", split="train")

    # Take a subset (shuffle for variety)
    dataset = dataset.shuffle(seed=42).select(range(min(args.num_samples * 3, len(dataset))))  # Extra samples in case of download failures

    print(f"Downloading up to {args.num_samples} images...")

    successful = []
    failed = 0

    def process_sample(idx, sample):
        key = sample['key']
        url = sample['url']
        caption = sample['caption']
        tags = sample.get('tags', '')

        save_path = os.path.join(images_dir, f"{key}.jpg")

        if download_image(url, save_path):
            # Use caption, or fall back to tags if caption is too short
            prompt = caption if len(caption) > 20 else f"{caption} {tags}".strip()
            # Truncate very long prompts
            if len(prompt) > 500:
                prompt = prompt[:500]
            return {
                'path': f"images/{key}.jpg",
                'prompt': prompt,
                'key': key
            }
        return None

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_sample, i, dataset[i]): i for i in range(len(dataset))}

        pbar = tqdm(total=args.num_samples, desc="Downloading")
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                successful.append(result)
                pbar.update(1)
                if len(successful) >= args.num_samples:
                    break
            else:
                failed += 1
        pbar.close()

    # Cancel remaining futures if we have enough
    for future in futures:
        future.cancel()

    print(f"\nDownloaded {len(successful)} images, {failed} failed")

    # Create metadata.csv
    df = pd.DataFrame(successful[:args.num_samples])
    metadata_path = os.path.join(args.output_dir, "metadata.csv")
    df.to_csv(metadata_path, index=False)
    print(f"Saved metadata to {metadata_path}")

    # Print sample
    print("\nSample entry:")
    print(df.head(1).to_string())


if __name__ == "__main__":
    main()
