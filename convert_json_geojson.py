import json
import os
import argparse
from tqdm import tqdm
import numpy as np

def convert_to_geojson(input_path, output_path):
    print(f"Loading {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Check structure
    if 'cells' in data:
        cells_list = data['cells']
        type_map = data.get('type_map', {})
        id_to_name = {}
        for k, v in type_map.items():
            # k is ID (str/int), v is Name (str)
            try:
                id_to_name[int(k)] = v
            except:
                pass
    else:
        print("Error: 'cells' key not found in JSON.")
        return

    # Define colors for standard CellViT classes (RGB)
    # Neoplastic: Red, Inflammatory: Green, Connective: Blue, Dead: Black, Epithelial: Orange
    color_map = {
        "Neoplastic": [255, 0, 0],
        "Inflammatory": [0, 255, 0],
        "Connective": [0, 0, 255],
        "Dead": [0, 0, 0],
        "Epithelial": [255, 159, 68],
        "Unknown": [128, 128, 128]
    }

    print(f"Converting {len(cells_list)} cells to QuPath-compatible GeoJSON List...")
    print(f"Writing to {output_path}...")

    with open(output_path, 'w') as f:
        f.write('[') # Start of list
        
        first = True
        for cell in tqdm(cells_list):
            contour = cell.get('contour')
            if not contour: 
                continue
                
            if contour[0] != contour[-1]:
                contour.append(contour[0])
                
            # Create Polygon geometry
            # QuPath working file had MultiPolygon: [[[x,y],...]]
            # But Polygon: [[x,y],...] is also valid.
            geometry = {
                "type": "Polygon",
                "coordinates": [contour]
            }
            
            cell_type_id = cell.get('type')
            cell_type_name = id_to_name.get(cell_type_id, str(cell_type_id))
            
            # QuPath properties
            properties = {
                "objectType": "annotation",
                "classification": {
                    "name": cell_type_name,
                    "color": color_map.get(cell_type_name, [128, 128, 128])
                },
                # "probability": cell.get('type_prob', 1.0) # QuPath might not use this in classification object
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
            
        f.write(']') # End of list
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert CellViT JSON to GeoJSON')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file')
    parser.add_argument('--output', type=str, required=True, help='Output GeoJSON file')
    args = parser.parse_args()

    convert_to_geojson(args.input, args.output)
