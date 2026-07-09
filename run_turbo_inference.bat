@echo off
echo Running Turbo Inference in Docker...
docker run --rm --gpus all --ipc=host ^
  --entrypoint /opt/conda/bin/python ^
  -v "%cd%\results":/results ^
  -v "%cd%\models":/models ^
  -v "%cd%":/workspace/CellViT-plus-plus ^
  ikimhoerst/cellvit:beta ^
  /workspace/CellViT-plus-plus/run_turbo_inference.py ^
  --input_dir /results/sampled_tiles_1024 ^
  --output_dir /results/sampled_tiles_1024_turbo_geojsons ^
  --model /models/CellViT-SAM-H-x40-AMP-001.pth ^
  --output_dir /results/sampled_tiles_1024_turbo_geojsons ^
  --model /models/CellViT-SAM-H-x40-AMP-001.pth ^
  --output_dir /results/sampled_tiles_1024_turbo_geojsons ^
  --model /models/CellViT-SAM-H-x40-AMP-001.pth ^
  --output_dir /results/sampled_tiles_1024_turbo_small_geojsons ^
  --model /models/CellViT-256-x40-AMP.pth ^
  --batch_size 8
