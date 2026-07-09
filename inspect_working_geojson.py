import json

input_path = "/results/small/TCGA-C8-A12V-01Z-00-DX1_cells.geojson"

print(f"Reading {input_path}...")
with open(input_path, 'r') as f:
    # It's a huge file, so we can't load it all.
    # We'll read until we find the first feature.
    buffer = ""
    while True:
        chunk = f.read(1024)
        if not chunk:
            break
        buffer += chunk
        # Look for the end of the first feature (approximate)
        if '"properties":' in buffer and '}' in buffer.split('"properties":')[1]:
             # Try to parse a bit more context
             try:
                 start_idx = buffer.find('{"type": "Feature"')
                 if start_idx != -1:
                     # This is rough, but we just want to see the keys
                     print("Header found.")
                     end_idx = buffer.find('}},', start_idx)
                     if end_idx != -1:
                         feature_str = buffer[start_idx:end_idx+2]
                         print(f"Sample Feature:\n{feature_str}")
                         break
             except:
                 pass
