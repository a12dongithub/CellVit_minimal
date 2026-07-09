import json
import sys

def inspect_json(path):
    print(f"Loading {path}...")
    with open(path, 'r') as f:
        data = json.load(f)
    
    print(f"Type: {type(data)}")
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        for k, v in data.items():
            print(f"Key '{k}': Type {type(v)}")
            if isinstance(v, list):
                print(f"  Length: {len(v)}")
                if len(v) > 0:
                     print(f"  First item type: {type(v[0])}")
                     if isinstance(v[0], dict):
                         print(f"  First item keys: {list(v[0].keys())}")
            elif isinstance(v, dict):
                print(f"  Length: {len(v)}")

inspect_json(sys.argv[1])
