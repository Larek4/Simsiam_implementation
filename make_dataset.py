import os
import xarray as xr
import numpy as np
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
ZARR_DIR = "data/ssl4eo-s12/train/S2L1C"     # Folder with .zarr.zip files
OUTPUT_DIR = "barlow_data/train"       # Output PNG images
BAND_INDICES = [3, 2, 1]  # Sentinel-2: RGB = B4, B3, B2 → 0-based indices
CLIP_RANGE = (0, 3000)
RESIZE = (224, 224)  # Output size for training (adjust if needed)

os.makedirs(OUTPUT_DIR, exist_ok=True)
zarr_files = sorted([f for f in os.listdir(ZARR_DIR) if f.endswith('.zarr')])

print(f"Found {len(zarr_files)} .zarr files in {ZARR_DIR}...")

for idx, fname in enumerate(tqdm(zarr_files)):
    zarr_path = os.path.join(ZARR_DIR, fname)

    try:
        ds = xr.open_zarr(zarr_path, consolidated=False)
        data = ds.bands.values  # Shape: [B, T, C, H, W]

        # Collapse B and T into single scenes
        for b in range(data.shape[0]):
            for t in range(data.shape[1]):
                scene = data[b, t]  # shape: [C, H, W]
                rgb = scene[BAND_INDICES]  # [3, H, W]

                # Clip and normalize
                rgb = np.clip(rgb, *CLIP_RANGE)
                rgb = (rgb - CLIP_RANGE[0]) / (CLIP_RANGE[1] - CLIP_RANGE[0]) * 255
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)

                # HWC format for PIL
                rgb_img = np.transpose(rgb, (1, 2, 0))
                image = Image.fromarray(rgb_img)

                if RESIZE:
                    image = image.resize(RESIZE)

                out_path = os.path.join(OUTPUT_DIR, f"scene_{idx:06d}_b{b}_t{t}.png")
                image.save(out_path)

    except Exception as e:
        print(f"Failed to process {fname}: {e}")

