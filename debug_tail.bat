@echo off
echo Reading tail of file...
docker run --rm ^
  --entrypoint /bin/bash ^
  -v "%cd%\results":/results ^
  ikimhoerst/cellvit:beta ^
  -c "tail -c 2000 /results/small/TCGA-C8-A12V-01Z-00-DX1_cells.geojson"
