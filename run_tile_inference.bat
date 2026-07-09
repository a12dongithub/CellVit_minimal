@echo off
echo Removing existing container if any...
docker rm -f cellvit-inference 2>nul

echo Running Tile Inference in Docker...
docker run --name cellvit-inference ^
  --rm ^
  --gpus all ^
  --entrypoint /opt/conda/bin/python ^
  -v "%cd%\models":/models ^
  -v "%cd%\results":/results ^
  -v "%cd%\cellvit":/workspace/CellViT-plus-plus/cellvit ^
  ikimhoerst/cellvit:beta ^
  cellvit/detect_cells.py ^
    --model /models/CellViT-SAM-H-x40-AMP-001.pth ^
    --outdir /results/sampled_tiles_1024_SAM_inference ^
    --gpu 0 ^
    process_dataset ^
    --wsi_folder /results/sampled_tiles_1024 ^
    --wsi_extension png
