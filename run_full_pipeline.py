import torch
import subprocess
import sys
import os
import argparse
import time
import importlib

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_dependencies():
    required = ['shapely', 'ujson', 'tqdm']
    for package in required:
        try:
            importlib.import_module(package)
        except ImportError:
            print(f"Installing missing dependency: {package}...")
            install(package)

def main():
    check_dependencies()

    parser = argparse.ArgumentParser(description="Run Full CellViT Pipeline (Inference + Splitting)")
    parser.add_argument("--input_dir", default="./data/Tiles", help="Input directory containing 1024x1024 PNG tiles")
    parser.add_argument("--output_base", default="./results", help="Base directory for outputs")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (Default 16, typically safe for H200)")
    parser.add_argument("--min_tumor_cells", type=int, default=2, help="Minimum tumor cells required to keep a tile (QC)")
    
    args = parser.parse_args()
    
    # 1. Detect GPUs
    if not torch.cuda.is_available():
        print("No CUDA GPUs found! Using CPU (Not recommended).")
        gpu_count = 0
    else:
        gpu_count = torch.cuda.device_count()
        print(f"Detected {gpu_count} GPUs.")
        
    pipeline_start = time.time()
    
    # Define intermediate directories
    intermediate_geojsons = os.path.join(args.output_base, "1024_predictions")
    final_output = os.path.join(args.output_base, "512_final_dataset")
    
    os.makedirs(intermediate_geojsons, exist_ok=True)
    os.makedirs(final_output, exist_ok=True)
    
    # 2. Step 1: Inference
    print("\n" + "="*50)
    print("STEP 1: Parallel Inference")
    print("="*50)
    
    inference_start = time.time()
    
    if gpu_count > 0:
        processes = []
        for rank in range(gpu_count):
            cmd = [
                sys.executable, "run_turbo_inference.py",
                "--input_dir", args.input_dir,
                "--output_dir", intermediate_geojsons,
                "--model", args.model,
                "--batch_size", str(args.batch_size),
                "--gpu", str(rank),
                "--rank", str(rank),
                "--world_size", str(gpu_count)
            ]
            print(f"Launching Worker {rank} on GPU {rank}...")
            p = subprocess.Popen(cmd)
            processes.append(p)
            
        for p in processes:
            p.wait()
            if p.returncode != 0:
                print("Error in inference worker! Exiting.")
                sys.exit(1)
    else:
        print("Running on CPU...")
        subprocess.run([
            sys.executable, "run_turbo_inference.py",
            "--input_dir", args.input_dir,
            "--output_dir", intermediate_geojsons,
            "--model", args.model,
            "--batch_size", "1",
            "--gpu", "0" 
        ], check=True)

    inference_end = time.time()
    inference_duration = inference_end - inference_start
    print(f"\n[BENCHMARK] Inference Time: {inference_duration:.2f} seconds ({inference_duration/60:.2f} minutes)")
    
    # 3. Step 2: Splitting + QC
    print("\n" + "="*50)
    print("STEP 2: Parallel Splitting + QC")
    print(f"Filter: Keeping tiles with >= {args.min_tumor_cells} tumor cells.")
    print("="*50)
    
    split_start = time.time()
    
    cmd_split = [
        sys.executable, "split_results.py",
        "--images", args.input_dir,
        "--geojsons", intermediate_geojsons,
        "--output", final_output,
        "--min_tumor_cells", str(args.min_tumor_cells)
    ]
    
    subprocess.run(cmd_split, check=True)
    
    split_end = time.time()
    split_duration = split_end - split_start
    
    total_duration = time.time() - pipeline_start
    
    print("\n" + "="*50)
    print(f"PIPELINE COMPLETE.")
    print(f"[BENCHMARK] Inference Time (GPU): {inference_duration:.2f}s ({inference_duration/60:.2f}m)")
    print(f"[BENCHMARK] Splitting + QC Time:  {split_duration:.2f}s ({split_duration/60:.2f}m)")
    print(f"[BENCHMARK] Total Pipeline Time:  {total_duration:.2f}s ({total_duration/60:.2f}m)")
    print(f"Final Dataset: {final_output}")
    print("="*50)

if __name__ == "__main__":
    main()
