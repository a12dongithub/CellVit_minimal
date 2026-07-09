@echo off
echo Removing existing container if any...
docker rm -f cellvit-inference 2>nul

echo Running Inference in Docker (SAM Model)...
docker run --name cellvit-inference ^
  --rm ^
  --gpus all ^
  --memory=32g ^
  --entrypoint /opt/conda/bin/python ^
  -v "%cd%\data":/data ^
  -v "%cd%\models":/models ^
  -v "%cd%\results":/results ^
  -v "%cd%\cellvit":/workspace/CellViT-plus-plus/cellvit ^
  ikimhoerst/cellvit:beta ^
  cellvit/detect_cells.py ^
    --model /models/CellViT-SAM-H-x40-AMP-001.pth ^
    --outdir /results ^
    --gpu 0 ^
    --batch_size 2 ^
    process_wsi ^
    --wsi_path /data/TCGA-C8-A12V-01Z-00-DX1.84B29360-B87B-4648-A697-B6610336C2BB.svs

echo Inference complete.
