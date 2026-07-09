import json
import sys

# Use the path directly or from arg
input_path = "/results/TCGA-C8-A12V-01Z-00-DX1_cells.json"

print(f"Loading {input_path} to read type_map...")
with open(input_path, 'r') as f:
    # We only need the beginning of the file where type_map usually lives, 
    # but since it's standard JSON, we might need to load it all or utilize a streaming parser.
    # Given we have 32GB RAM in container (hopefully), loading it should be fine as done before.
    # Alternatively, we can use ijson but simplest is just load and print data['type_map']
    data = json.load(f)

if 'type_map' in data:
    print("Found type_map:")
    print(json.dumps(data['type_map'], indent=2))
else:
    print("type_map key NOT found.")
