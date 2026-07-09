#!/bin/bash
set -e

# ==============================================================================
# CellViT-Plus-Plus Cloud Pipeline (Self-Contained, Native)
# ==============================================================================
# 1. Installs PyTorch (cu128) + CuPy (cuda12x)
# 2. Generates the correct run_turbo_inference.py (with Ray disabled)
# 3. Runs the full pipeline
# ==============================================================================

echo "============================================================"
echo "   CellViT Cloud Pipeline"
echo "============================================================"

# --- 1. Detect Python ---
PY=python3
if command -v python3 &> /dev/null; then
    PY=python3
elif command -v python &> /dev/null; then
    PY=python
fi
echo "Using Python: $PY ($($PY --version 2>&1))"

# --- 2. GPU Check ---
echo ""
echo "[1/6] Checking GPU Environment..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. GPU may not be available."
fi

# --- 3. Overwrite run_turbo_inference.py (Fix Ray Hang) ---
echo ""
echo "[2/6] Generating fixed inference script (Disabling Ray)..."
cat << 'EOF' > run_turbo_inference.py
import sys
from unittest.mock import MagicMock

# CRITICAL FIX: Mock Ray to prevent "Started a local Ray instance" hang
# We cannot use sys.modules["ray"] = None because some versions of the repo
# import ray without a try-except block, causing ModuleNotFoundError.
# Instead, we inject a Mock object that acts as a pass-through.

mock_ray = MagicMock()

# ray.remote(...) returns a decorator. The decorator returns the class/func unmodified.
def remote_decorator_shim(*args, **kwargs):
    def decorator(cls_or_func):
        return cls_or_func
    return decorator

# ray.init() should do nothing
def init_shim(*args, **kwargs):
    pass

mock_ray.remote.side_effect = remote_decorator_shim
mock_ray.init.side_effect = init_shim

sys.modules["ray"] = mock_ray

import argparse
import os
import time
import importlib
import glob
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image, PngImagePlugin, ImageFile
from tqdm import tqdm
from pathlib import Path
import numpy as np
import uuid

# Increase PIL limits for large files/metadata
PngImagePlugin.MAX_TEXT_CHUNK = 200 * (1024**2) 
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import cupy as cp

# Add cellvit path
sys.path.append("/workspace/CellViT-plus-plus")
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from cellvit.inference.postprocessing_cupy import DetectionCellPostProcessorCupy
from cellvit.config.config import TYPE_NUCLEI_DICT_PANNUKE, COLOR_DICT_CELLS
from cellvit.config.templates import get_template_segmentation
from cellvit.utils.tools import unflatten_dict
from cellvit.models.cell_segmentation.cellvit import CellViT
from cellvit.models.cell_segmentation.cellvit_256 import CellViT256
from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM
from cellvit.models.cell_segmentation.cellvit_uni import CellViTUNI

# Mock WSI object to satisfy DetectionCellPostProcessorCupy
class MockWSI:
    def __init__(self, patch_size=1024):
        self.metadata = {
            "patch_size": patch_size,
            "patch_overlap": 0,
            "target_patch_mpp": 0.25,
            "base_mpp": 0.25,
            "downsampling": 1
        }

class TileDataset(Dataset):
    def __init__(self, image_dir, transform=None, rank=0, world_size=1):
        all_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        if world_size > 1:
            self.image_paths = all_files[rank::world_size]
        else:
            self.image_paths = all_files
        print(f"Worker {rank}/{world_size}: Processing {len(self.image_paths)}/{len(all_files)} tiles.")
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, Path(path).stem

def get_transform():
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    return T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])

def save_geojson(cells_dict, output_path, label_map):
    features = []
    for cell_id, cell_data in cells_dict.items():
        cell_type = cell_data["type"]
        if cell_type == 0: continue 
        contour = cell_data["contour"]
        if isinstance(contour, np.ndarray):
            contour = contour.tolist()
        if not contour: continue
        if contour[0] != contour[-1]:
            contour.append(contour[0])
            
        type_name = label_map.get(cell_type, "Unknown")
        color = COLOR_DICT_CELLS.get(cell_type, [128, 128, 128])
        
        feature = {
            "type": "Feature",
            "id": str(uuid.uuid4()),
            "geometry": {
                "type": "Polygon",
                "coordinates": [contour]
            },
            "properties": {
                "objectType": "annotation",
                "classification": {
                    "name": type_name,
                    "color": color
                }
            }
        }
        features.append(feature)
        
    with open(output_path, 'w') as f:
        json.dump(features, f)

def load_model_local(model_path, device):
    print(f"Loading model from: {model_path}")
    model_checkpoint = torch.load(model_path, map_location="cpu")
    run_conf = unflatten_dict(model_checkpoint["config"], ".")
    model_arch = model_checkpoint["arch"]
    print(f"Detected architecture: {model_arch}")
    
    if model_arch == "CellViT":
        model = CellViT(
            num_nuclei_classes=run_conf["data"]["num_nuclei_classes"],
            num_tissue_classes=run_conf["data"]["num_tissue_classes"],
            embed_dim=run_conf["model"]["embed_dim"],
            input_channels=run_conf["model"].get("input_channels", 3),
            depth=run_conf["model"]["depth"],
            num_heads=run_conf["model"]["num_heads"],
            extract_layers=run_conf["model"]["extract_layers"],
            regression_loss=run_conf["model"].get("regression_loss", False),
        )
    elif model_arch == "CellViT256":
        model = CellViT256(
            model256_path=None,
            num_nuclei_classes=run_conf["data"]["num_nuclei_classes"],
            num_tissue_classes=run_conf["data"]["num_tissue_classes"],
            regression_loss=run_conf["model"].get("regression_loss", False),
        )
    elif model_arch == "CellViTSAM":
        model = CellViTSAM(
            model_path=None,
            num_nuclei_classes=run_conf["data"]["num_nuclei_classes"],
            num_tissue_classes=run_conf["data"]["num_tissue_classes"],
            vit_structure=run_conf["model"]["backbone"],
            regression_loss=run_conf["model"].get("regression_loss", False),
        )
    elif model_arch == "CellViTUNI":
        model = CellViTUNI(
            model_uni_path=None,
            num_nuclei_classes=run_conf["data"]["num_nuclei_classes"],
            num_tissue_classes=run_conf["data"]["num_tissue_classes"],
        )
    else:
        raise NotImplementedError(f"Unknown model architecture: {model_arch}")

    model.load_state_dict(model_checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    
    if hasattr(model, "patch_size"):
        run_conf["model"]["token_patch_size"] = model.patch_size

    mixed_precision = run_conf["training"].get("mixed_precision", False)
    return model, run_conf, model_arch, mixed_precision

def main():
    parser = argparse.ArgumentParser(description="Turbo CellViT Inference")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading model...")
    model, run_conf, model_arch, use_amp = load_model_local(args.model, device)
    use_amp = True 
    
    print("Setting up post-processor...")
    mock_wsi = MockWSI()
    post_processor = DetectionCellPostProcessorCupy(
        wsi=mock_wsi,
        nr_types=run_conf["data"]["num_nuclei_classes"],
        resolution=0.25,
        binary=False
    )
    
    print("Creating DataLoader...")
    dataset = TileDataset(args.input_dir, transform=get_transform(), rank=args.rank, world_size=args.world_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    label_map = TYPE_NUCLEI_DICT_PANNUKE 
    
    print(f"Starting inference on {len(dataset)} tiles...")
    model.eval()
    with torch.no_grad():
        for images, filenames in tqdm(dataloader):
            images = images.to(device)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    predictions = model.forward(images, retrieve_tokens=True)
            else:
                predictions = model.forward(images, retrieve_tokens=True)
                
            predictions["nuclei_binary_map"] = F.softmax(predictions["nuclei_binary_map"], dim=1)
            predictions["nuclei_type_map"] = F.softmax(predictions["nuclei_type_map"], dim=1)
            predictions["nuclei_type_map"] = predictions["nuclei_type_map"].permute(0, 2, 3, 1)
            predictions["nuclei_binary_map"] = predictions["nuclei_binary_map"].permute(0, 2, 3, 1)
            predictions["hv_map"] = predictions["hv_map"].permute(0, 2, 3, 1)
            
            try:
                _, cell_dicts_list = post_processor.post_process_batch(predictions)
                for i, cell_dict in enumerate(cell_dicts_list):
                    fname = filenames[i]
                    out_path = os.path.join(args.output_dir, f"{fname}.geojson")
                    save_geojson(cell_dict, out_path, label_map)
            except Exception as e:
                print(f"Error processing batch: {e}")
                
    print("Inference completed.")

if __name__ == "__main__":
    main()
EOF

# --- 4. Install Dependencies ---
echo ""
echo "[3/6] Installing PyTorch (cu128) + CuPy..."
NEEDS_SETUP=true
$PY -c "import torch, cupy; assert torch.cuda.is_available()" 2>/dev/null && NEEDS_SETUP=false

if [ "$NEEDS_SETUP" = true ]; then
    echo "Setting up PyTorch and CuPy from scratch..."
    $PY -m pip uninstall -y cupy cupy-cuda11x cupy-cuda12x cupy-cuda13x torch torchvision torchaudio nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 2>/dev/null || true
    $PY -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
    $PY -m pip install "cupy-cuda12x>=13.0.0"
else
    echo "PyTorch and CuPy already installed."
fi

echo ""
echo "[4/6] Installing other dependencies..."
# Added 'ray' here just in case, but disabled in Python script
$PY -m pip install einops timm shapely ujson tqdm scikit-image scipy matplotlib pandas colorama "numpy<2.0" numba Pillow ray opencv-python-headless 2>/dev/null || true

# --- 5. Verify Environment ---
echo ""
echo "[5/6] Verifying Environment..."

# FIX: Add all nvidia libraries to LD_LIBRARY_PATH
# This resolves "Unable to load libcudnn_graph.so" and "Cannot load symbol cudnnGetVersion"
echo "Auto-configuring LD_LIBRARY_PATH for NVIDIA libraries..."
export LD_LIBRARY_PATH=$($PY -c "import os, site; print(':'.join([os.path.join(p, 'nvidia', d, 'lib') for p in site.getsitepackages() for d in os.listdir(os.path.join(p, 'nvidia')) if os.path.exists(os.path.join(p, 'nvidia', d, 'lib'))] + [os.environ.get('LD_LIBRARY_PATH', '')]))")
echo "LD_LIBRARY_PATH updated."

$PY -c "
import torch, cupy
print(f'  PyTorch: {torch.__version__} (CUDA: {torch.version.cuda})')
print(f'  CuPy:    {cupy.__version__}')
assert torch.cuda.is_available(), 'CUDA not available!'
print('  All checks passed.')
" || exit 1

# --- 6. Run Pipeline ---
echo ""
echo "[6/6] Launching Pipeline..."

# Verify Data
if [ ! -d "./data/Tiles" ]; then
    echo "ERROR: ./data/Tiles directory not found!"
    exit 1
fi
TILE_COUNT=$(ls ./data/Tiles/*.png 2>/dev/null | wc -l)
echo "Found $TILE_COUNT tiles in ./data/Tiles"

if [ ! -f "./run_full_pipeline.py" ]; then
    echo "ERROR: run_full_pipeline.py not found!"
    exit 1
fi

MODEL_PATH="./models/CellViT-SAM-H-x40-AMP-001.pth"
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

export PYTHONPATH="$(pwd):$PYTHONPATH"
$PY run_full_pipeline.py \
    --input_dir ./data/Tiles \
    --output_base ./results \
    --model $MODEL_PATH \
    --batch_size 64 \
    --min_tumor_cells 2

echo "============================================================"
echo "Pipeline Complete! Check ./results"
echo "============================================================"
