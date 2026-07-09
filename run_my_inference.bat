@echo off
docker rm -f cellvit-inference 2>nul
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
    --outdir /results/Adipokines_TMA4 ^
    --gpu 0 ^
    --geojson ^
    --batch_size 2 ^
    process_wsi ^
    --wsi_path /data/Adipokines_TMA4_2025.2.21.svs
