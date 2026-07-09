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
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file_path = log_dir / f"{svs_path.name}.log"
    
    print(f"[{svs_path.name}] Starting on GPU {gpu_id}... (tail -f {log_file_path} for progress)")
    try:
        with open(log_file_path, "w") as log_file:
            subprocess.run(cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT)
        print(f"[{svs_path.name}] Successfully completed on GPU {gpu_id}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[{svs_path.name}] Error processing! Command exited with {e.returncode}. See {log_file_path} for details.")
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

    # Process files concurrently
    success_count = 0
    with ThreadPoolExecutor(max_workers=args.num_gpus) as executor:
        futures = {executor.submit(worker, svs_path, gpu_queue, args): svs_path for svs_path in svs_files}
        
        for future in as_completed(futures):
            svs_path = futures[future]
            try:
                if future.result():
                    success_count += 1
            except Exception as exc:
                print(f"[{svs_path.name}] Generated an exception: {exc}")

    print(f"\n============================================================")
    print(f" Batch processing complete.")
    print(f" Successfully processed {success_count}/{len(svs_files)} files.")
    print(f"============================================================")

if __name__ == "__main__":
    main()

