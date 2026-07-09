import json
import os
import glob
from tqdm import tqdm
from pathlib import Path

def convert_to_geojson(input_path, output_path):
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)

        if 'cells' not in data:
            print(f"Skipping {input_path}: 'cells' key not found.")
            return False

        cells_list = data['cells']
        type_map = data.get('type_map', {})
        
        id_to_name = {}
        for k, v in type_map.items():
            try:
                id_to_name[int(k)] = v
            except:
                pass

        # Color map
        color_map = {
            "Neoplastic": [255, 0, 0],
            "Inflammatory": [0, 255, 0],
            "Connective": [0, 0, 255],
            "Dead": [0, 0, 0],
            "Epithelial": [255, 159, 68],
            "Unknown": [128, 128, 128]
        }

        with open(output_path, 'w') as f:
            f.write('[')
            first = True
            for cell in cells_list:
                contour = cell.get('contour')
                if not contour: 
                    continue
                
                if contour[0] != contour[-1]:
                    contour.append(contour[0])
                    
                geometry = {
                    "type": "Polygon",
                    "coordinates": [contour]
                }
                
                cell_type_id = cell.get('type')
                cell_type_name = id_to_name.get(cell_type_id, str(cell_type_id))
                
                properties = {
                    "objectType": "annotation",
                    "classification": {
                        "name": cell_type_name,
                        "color": color_map.get(cell_type_name, [128, 128, 128])
                    }
                }
                
                feature = {
                    "type": "Feature",
                    "id": str(cell.get('id', '')),
                    "geometry": geometry,
                    "properties": properties
                }
                
                if not first:
                    f.write(',')
                else:
                    first = False
                
                json.dump(feature, f)
            f.write(']')
        return True
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return False

def main():
    BASE_DIR = "/results/sampled_tiles_1024_SAM_inference"
    OUTPUT_DIR = "/results/sampled_tiles_1024_SAM_geojsons"
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Find all cells.json files
    search_pattern = os.path.join(BASE_DIR, "**", "*cells.json")
    files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(files)} result files.")
    
    for input_path in tqdm(files):
        # Construct output filename from input filename
        # Input: tile_XXX_..._cells.json
        # Output: tile_XXX_... .geojson
        input_filename = Path(input_path).name
        if input_filename.endswith("_cells.json"):
            tile_name = input_filename.replace("_cells.json", "")
        else:
            tile_name = Path(input_path).stem
            
        output_filename = f"{tile_name}.geojson"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        convert_to_geojson(input_path, output_path)
        
    print("Batch conversion completed.")

if __name__ == "__main__":
    main()
