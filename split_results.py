import os
import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
from shapely.geometry import shape, Point, Polygon
import shutil

def setup_directories(output_base):
    img_out = os.path.join(output_base, "images")
    json_out = os.path.join(output_base, "geojsons")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(json_out, exist_ok=True)
    return img_out, json_out

def split_image(image_path, output_dir, valid_quadrants=None):
    try:
        img = Image.open(image_path)
        filename = Path(image_path).stem
        
        crops = {
            "TL": (0, 0, 512, 512),
            "TR": (512, 0, 1024, 512),
            "BL": (0, 512, 512, 1024),
            "BR": (512, 512, 1024, 1024)
        }
        
        for suffix, box in crops.items():
            # If valid_quadrants is specified, skip invalid ones
            if valid_quadrants is not None and suffix not in valid_quadrants:
                continue
                
            crop = img.crop(box)
            out_name = f"{filename}_{suffix}.png"
            crop.save(os.path.join(output_dir, out_name))
            
        return True
    except Exception as e:
        print(f"Error splitting image {image_path}: {e}")
        return False

def analyze_and_split_geojson(geojson_path, image_name, min_tumor_cells=0):
    try:
        with open(geojson_path, 'r') as f:
            features = json.load(f)
            
        quadrants = {
            "TL": [], "TR": [], "BL": [], "BR": []
        }
        
        offsets = {
            "TL": (0, 0), "TR": (512, 0), "BL": (0, 512), "BR": (512, 512)
        }
        
        bounds = {
            "TL": (0, 0, 512, 512), "TR": (512, 0, 1024, 512),
            "BL": (0, 512, 512, 1024), "BR": (512, 512, 1024, 1024)
        }
        
        for feature in features:
            geom = shape(feature['geometry'])
            centroid = geom.centroid
            cx, cy = centroid.x, centroid.y
            
            target_quad = None
            for q_name, (min_x, min_y, max_x, max_y) in bounds.items():
                if min_x <= cx < max_x and min_y <= cy < max_y:
                    target_quad = q_name
                    break
            
            if target_quad:
                # Coordinate Translation
                off_x, off_y = offsets[target_quad]
                
                def translate_coords(coords, dx, dy):
                    new_coords = []
                    for item in coords:
                        if isinstance(item[0], (list, tuple)):
                             new_coords.append(translate_coords(item, dx, dy))
                        else:
                             new_coords.append([item[0] - dx, item[1] - dy])
                    return new_coords

                new_coords = translate_coords(feature['geometry']['coordinates'], off_x, off_y)
                
                new_feature = feature.copy()
                new_feature['geometry'] = {
                    "type": feature['geometry']['type'],
                    "coordinates": new_coords
                }
                quadrants[target_quad].append(new_feature)
        
        # Filter (Don't save yet)
        valid_data = {}
        
        for q_name, feature_list in quadrants.items():
            if not feature_list:
                continue

            # QC Check: Count Tumor Cells
            tumor_count = 0
            for feat in feature_list:
                props = feat.get("properties", {})
                cls = props.get("classification", {})
                name = cls.get("name", "Unknown")
                
                # Check for Neoplastic/Tumor
                if name == "Neoplastic" or name == "Tumor":
                    tumor_count += 1
            
            if tumor_count >= min_tumor_cells:
                valid_data[q_name] = feature_list
                
        return valid_data

    except Exception as e:
        print(f"Error splitting geojson {geojson_path}: {e}")
        return {}

def process_tile(args_tuple):
    # Unpack
    img_path, geojson_in_dir, img_out_dir, json_out_dir, min_tumor_cells = args_tuple
    
    img_name = Path(img_path).name
    basename = Path(img_path).stem
    
    # 1. Check GeoJSON first (for QC)
    geojson_path = os.path.join(geojson_in_dir, f"{basename}.geojson")
    
    if os.path.exists(geojson_path):
        # Analyze and split GeoJSON, returning data for quadrants that pass QC
        valid_data = analyze_and_split_geojson(
            geojson_path, img_name, min_tumor_cells
        )
        
        if valid_data:
            # 2. Split Image (Only for valid quadrants)
            # We assume split_image returns True if ALL requested quadrants were saved successfully.
            if split_image(img_path, img_out_dir, valid_data.keys()):
                # 3. Save GeoJSONs (Only IF image split succeeded)
                try:
                    for q_name, feature_list in valid_data.items():
                        out_name = f"{basename}_{q_name}.geojson"
                        out_path = os.path.join(json_out_dir, out_name)
                        with open(out_path, 'w') as f:
                            json.dump(feature_list, f)
                    return True
                except Exception as e:
                    print(f"Error saving geojsons for {basename}: {e}")
                    return False
            else:
                return False # Image split failed
        else:
            return False # No quadrants passed QC
    else:
        # No GeoJSON -> No tumor cells -> QC Fail -> Skip Image
        return False

def main():
    parser = argparse.ArgumentParser(description="Split tiles and GeoJSONs into 4 quadrants (Parallel) with QC")
    parser.add_argument("--images", required=True, help="Input directory for 1024x1024 images")
    parser.add_argument("--geojsons", required=True, help="Input directory for corresponding GeoJSONs")
    parser.add_argument("--output", required=True, help="Output base directory")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of worker processes")
    parser.add_argument("--min_tumor_cells", type=int, default=0, help="Minimum tumor cells required to keep a tile")
    
    args = parser.parse_args()
    
    img_out_dir, json_out_dir = setup_directories(args.output)
    
    import glob
    from concurrent.futures import ProcessPoolExecutor
    
    image_files = glob.glob(os.path.join(args.images, "*.png"))
    print(f"Found {len(image_files)} images.")
    print(f"Running filtering with Min Tumor Cells: {args.min_tumor_cells}")
    
    # Pass min_tumor_cells to worker
    tasks = [(f, args.geojsons, img_out_dir, json_out_dir, args.min_tumor_cells) for f in image_files]
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(executor.map(process_tile, tasks), total=len(tasks)))
        
    saved_count = sum(results)
    print(f"Splitting completed. Saved {saved_count} tiles (filtered from {len(results)} originals).")

if __name__ == "__main__":
    main()
