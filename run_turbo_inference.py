import argparse
import os
import sys
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
PngImagePlugin.MAX_TEXT_CHUNK = 200 * (1024**2)  # 200MB Limit
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
            "patch_overlap": 0, # Irrelevant for local processing
            "target_patch_mpp": 0.25,
            "base_mpp": 0.25,
            "downsampling": 1
        }

class TileDataset(Dataset):
    def __init__(self, image_dir, transform=None, rank=0, world_size=1):
        all_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        
        if world_size > 1:
            # Simple striding for sharding
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
    # Matches _load_inference_transforms in CellViTInference
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    return T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])

def save_geojson(cells_dict, output_path, label_map):
    # Convert cell_dict (from post_process_single_image) to GeoJSON
    # cells_dict keys: "bbox", "centroid", "contour", "type_prob", "type"
    
    features = []
    
    # Values in cells_dict are sub-dictionaries for each cell instance
    for cell_id, cell_data in cells_dict.items():
        cell_type = cell_data["type"]
        if cell_type == 0: continue # Skip background
        
        contour = cell_data["contour"]
        # Contour is likely (N, 2) numpy array. Convert to closed polygon list of lists
        if isinstance(contour, np.ndarray):
            contour = contour.tolist()
            
        if not contour: continue
        
        # Close polygon
        if contour[0] != contour[-1]:
            contour.append(contour[0])
            
        # Create GeoJSON feature
        
        type_name = label_map.get(cell_type, "Unknown")
        # Fix: Use integer cell_type for color lookup
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
    
    # Instantiate correct model class
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

    # Load state dict
    model.load_state_dict(model_checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    
    # Set patch size in config if needed
    if hasattr(model, "patch_size"):
        run_conf["model"]["token_patch_size"] = model.patch_size

    # AMP setting
    mixed_precision = run_conf["training"].get("mixed_precision", False)
    
    return model, run_conf, model_arch, mixed_precision

def main():
    parser = argparse.ArgumentParser(description="Turbo CellViT Inference")
    parser.add_argument("--input_dir", required=True, help="Directory containing 1024x1024 PNG tiles")
    parser.add_argument("--output_dir", required=True, help="Output directory for GeoJSONs")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--rank", type=int, default=0, help="Shard rank (0-indexed)")
    parser.add_argument("--world_size", type=int, default=1, help="Total number of shards")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Model
    print("Loading model...")
    model, run_conf, model_arch, use_amp = load_model_local(args.model, device)
    
    # Force AMP if desired (CellViTInference bad this as arg, we can default true or check config)
    # The original script enforced it via argument enforce_mixed_precision=True
    use_amp = True 
    
    # 2. Setup Post-Processor
    print("Setting up post-processor...")
    mock_wsi = MockWSI()
    # run_conf["data"]["num_nuclei_classes"] usually includes background
    post_processor = DetectionCellPostProcessorCupy(
        wsi=mock_wsi,
        nr_types=run_conf["data"]["num_nuclei_classes"],
        resolution=0.25,
        binary=False
    )
    
    # 3. Data Loader
    print("Creating DataLoader...")
    dataset = TileDataset(
        args.input_dir, 
        transform=get_transform(), 
        rank=args.rank, 
        world_size=args.world_size
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    label_map = TYPE_NUCLEI_DICT_PANNUKE # Default for this project
    
    print(f"Starting inference on {len(dataset)} tiles...")
    
    model.eval()
        
    with torch.no_grad():
        for images, filenames in tqdm(dataloader):
            images = images.to(device)
            
            # Forward Pass
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    predictions = model.forward(images, retrieve_tokens=True)
            else:
                predictions = model.forward(images, retrieve_tokens=True)
                
            # Reorder & Softmax (Logic from CellViTInference.apply_softmax_reorder)
            predictions["nuclei_binary_map"] = F.softmax(predictions["nuclei_binary_map"], dim=1)
            predictions["nuclei_type_map"] = F.softmax(predictions["nuclei_type_map"], dim=1)
            predictions["nuclei_type_map"] = predictions["nuclei_type_map"].permute(0, 2, 3, 1)
            predictions["nuclei_binary_map"] = predictions["nuclei_binary_map"].permute(0, 2, 3, 1)
            predictions["hv_map"] = predictions["hv_map"].permute(0, 2, 3, 1)
            
            # Post-Process Batch
            try:
                # post_process_batch returns (instance_map_tensor, list_of_cell_dicts)
                # We only need the list of cell dicts
                _, cell_dicts_list = post_processor.post_process_batch(predictions)
                
                # Save Results
                for i, cell_dict in enumerate(cell_dicts_list):
                    fname = filenames[i]
                    out_path = os.path.join(args.output_dir, f"{fname}.geojson")
                    save_geojson(cell_dict, out_path, label_map)
                    
            except Exception as e:
                print(f"Error processing batch: {e}")
                
    print("Inference completed.")

if __name__ == "__main__":
    main()
