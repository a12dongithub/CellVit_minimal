import json
import sys

input_path = "/results/TCGA-C8-A12V-01Z-00-DX1_cells_fast.geojson"
output_path = "/results/sample_cells.geojson"

print(f"Reading {input_path}...")
# Read first few bytes to check header
with open(input_path, 'r') as f:
    header = f.read(100)
    print(f"Header: {header}")

print("Parsing full file to extract sample...")
with open(input_path, 'r') as f:
    data = json.load(f)

print(f"Total features: {len(data)}")

# New format is a list of features
sample = data[:100]

print(f"Writing sample to {output_path}...")
with open(output_path, 'w') as f:
    json.dump(sample, f, indent=2)

print("Done.")
