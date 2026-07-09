@echo off
call .venv\Scripts\activate

echo Running Inference...
python cellvit/detect_cells.py ^
    --model ./models/CellViT-256-x40-AMP.pth ^
    --wsi_path ./data/TCGA-C8-A12V-01Z-00-DX1.84B29360-B87B-4648-A697-B6610336C2BB.svs ^
    --outdir ./results ^
    --gpu 0 ^
    --geojson ^
    --enforce_amp ^
    process_wsi ^
    --wsi_path ./data/TCGA-C8-A12V-01Z-00-DX1.84B29360-B87B-4648-A697-B6610336C2BB.svs

echo Inference complete.
