import openslide
import numpy as np
from PIL import Image
import os
import random
from tqdm import tqdm

# Configuration
WSI_PATH = "/data/TCGA-C8-A12V-01Z-00-DX1.84B29360-B87B-4648-A697-B6610336C2BB.svs"
OUTPUT_DIR = "/results/sampled_tiles_1024"
TILE_SIZE = 1024
NUM_TILES = 10
TISSUE_THRESHOLD = 235 # Pixel intensity threshold (lower is darker/tissue)
MAX_ATTEMPTS = 5000

def is_tissue(tile_np):
    # Simple check: Convert to grayscale
    # If mean intensity is high (white), it's background.
    # Also check if it's not completely black (artifact)
    gray = np.mean(tile_np, axis=2)
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    
    # Criteria: 
    # 1. Not too bright (background is usually ~240-255)
    # 2. Not too dark (artifacts)
    # 3. Has variance (tissue has texture, background is flat)
    return mean_val < TISSUE_THRESHOLD and mean_val > 40 and std_val > 5

def sample_tiles():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Opening WSI: {WSI_PATH}")
    slide = openslide.OpenSlide(WSI_PATH)
    w, h = slide.dimensions
    print(f"Dimensions: {w}x{h}")
    
    count = 0
    attempts = 0
    
    pbar = tqdm(total=NUM_TILES)
    
    while count < NUM_TILES and attempts < MAX_ATTEMPTS:
        attempts += 1
        x = random.randint(0, w - TILE_SIZE)
        y = random.randint(0, h - TILE_SIZE)
        
        try:
            tile = slide.read_region((x, y), 0, (TILE_SIZE, TILE_SIZE))
            tile = tile.convert("RGB")
            tile_np = np.array(tile)
            
            if is_tissue(tile_np):
                filename = f"tile_{count:03d}_x{x}_y{y}.png"
                tile.save(os.path.join(OUTPUT_DIR, filename))
                count += 1
                pbar.update(1)
        except Exception as e:
            print(f"Error reading region: {e}")
            continue
            
    pbar.close()
    if count < NUM_TILES:
        print(f"Warning: Only collected {count} tiles after {attempts} attempts.")
    else:
        print(f"Successfully collected {count} tiles.")

if __name__ == "__main__":
    sample_tiles()
