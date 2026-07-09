import os
import sys
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

def process_single_svs(svs_path, gpu_id, args):
    svs_parent = svs_path.parent
    
    # Check if this slide has already been processed
    expected_output_1 = svs_parent / f"{svs_path.stem}_cells.geojson"
    expected_output_2 = svs_parent / f"{svs_path.stem}_cell_detection.geojson"
    
    if expected_output_1.exists() or expected_output_2.exists():
        print(f"[{svs_path.name}] Skipping. Found existing GeoJSON output in {svs_parent}")
        return True
    
    cmd = [
        sys.executable, "-m", "cellvit.detect_cells",
        "--model", args.model,
        "--outdir", str(svs_parent),
        "--gpu", str(gpu_id),
        "--batch_size", str(args.batch_size),
        "--resolution", str(args.resolution),
        "--geojson",
        "process_wsi",
        "--wsi_path", str(svs_path)
    ]
    
    print(f"[{svs_path.name}] Starting on GPU {gpu_id}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[{svs_path.name}] Successfully completed on GPU {gpu_id}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[{svs_path.name}] Error processing! Command exited with {e.returncode}")
        print(f"[{svs_path.name}] --- ERROR OUTPUT ---")
        print(e.stderr if e.stderr else e.stdout)
        print(f"[{svs_path.name}] ----------------------")
        return False

def worker(svs_path, gpu_queue, args):
    # Get an available GPU
    gpu_id = gpu_queue.get()
    try:
        success = process_single_svs(svs_path, gpu_id, args)
        return success
    finally:
        # Always return the GPU to the queue when done
        gpu_queue.put(gpu_id)

def main():
    parser = argparse.ArgumentParser(description="Batch process WSIs in nested folders using CellViT")
    parser.add_argument("--data_dir", default="data", help="Path to the root data directory containing nested SVS files")
    parser.add_argument("--model", default="models/CellViT-SAM-H-x40-AMP-001.pth", help="Path to model checkpoint (.pth)")
    parser.add_argument("--num_gpus", type=int, default=1, help="Total number of GPUs available for parallel processing")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for model inference. Default: 8")
    parser.add_argument("--resolution", type=float, default=0.25, help="Resolution (MPP) for inference. Default: 0.25")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"Error: Data directory '{data_path}' does not exist.")
        sys.exit(1)

    # Recursively find all .svs files
    svs_files = list(data_path.rglob("*.svs"))
    
    if not svs_files:
        print(f"No .svs files found in {data_path}")
        sys.exit(0)

    print(f"============================================================")
    print(f" Found {len(svs_files)} SVS files in {data_path}")
    print(f" Utilizing {args.num_gpus} GPUs concurrently")
    print(f" Batch Size: {args.batch_size}")
    print(f"============================================================")
    
    # Initialize a queue with the available GPU IDs
    gpu_queue = queue.Queue()
    for i in range(args.num_gpus):
        gpu_queue.put(i)

    print("Starting Global Ray Cluster to prevent worker conflicts...")
    subprocess.run([sys.executable, "-m", "ray.scripts.scripts", "stop"], capture_output=True)
    subprocess.run([sys.executable, "-m", "ray.scripts.scripts", "start", "--head", "--port=6379", "--include-dashboard=false"], check=True)
    os.environ["RAY_ADDRESS"] = "127.0.0.1:6379"

    # Process files concurrently
    success_count = 0
    try:
        with ThreadPoolExecutor(max_workers=args.num_gpus) as executor:
            futures = {executor.submit(worker, svs_path, gpu_queue, args): svs_path for svs_path in svs_files}
            
            for future in as_completed(futures):
                svs_path = futures[future]
                try:
                    if future.result():
                        success_count += 1
                except Exception as exc:
                    print(f"[{svs_path.name}] Generated an exception: {exc}")
    finally:
        print("Shutting down Global Ray Cluster...")
        subprocess.run([sys.executable, "-m", "ray.scripts.scripts", "stop"], capture_output=True)

    print(f"\n============================================================")
    print(f" Batch processing complete.")
    print(f" Successfully processed {success_count}/{len(svs_files)} files.")
    print(f"============================================================")

if __name__ == "__main__":
    main()

