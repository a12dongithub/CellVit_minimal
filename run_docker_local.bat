@echo off
REM =========================================================
REM Run CellViT Pipeline Locally using Docker (Windows)
REM Requires Docker Desktop installed and running.
REM =========================================================

echo Starting CellViT Docker Pipeline...
echo Ensure Docker Desktop is running!

docker run --rm ^
    --gpus all ^
    --shm-size=32g ^
    -v "%cd%":/workspace/context ^
    -w /workspace/context ^
    ikimhoerst/cellvit:beta ^
    python run_full_pipeline.py ^
    --input_dir ./data/Tiles ^
    --output_base ./results ^
    --model /workspace/context/models/CellViT-SAM-H-x40-AMP-001.pth ^
    --batch_size 4 ^
    --min_tumor_cells 2

echo Pipeline finished! Check ./results folder.
pause
