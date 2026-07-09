@echo off
echo Reading head of file...
docker run --rm ^
  --entrypoint /bin/bash ^
  -v "%cd%\results":/results ^
  ikimhoerst/cellvit:beta ^
  -c "dd if=/results/small/TCGA-C8-A12V-01Z-00-DX1_cells.geojson bs=2000 count=1 2>/dev/null"
