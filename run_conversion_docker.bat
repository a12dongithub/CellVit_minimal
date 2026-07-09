@echo off
echo Removing existing container if any...
docker rm -f cellvit-inference 2>nul

echo Running Conversion in Docker...
docker run --name cellvit-inference ^
  --rm ^
  --entrypoint /opt/conda/bin/python ^
  -v "%cd%\results":/results ^
  -v "%cd%\convert_json_geojson.py":/workspace/convert_json_geojson.py ^
  ikimhoerst/cellvit:beta ^
  /workspace/convert_json_geojson.py ^
    --input /results/TCGA-C8-A12V-01Z-00-DX1_cells.json ^
    --output /results/TCGA-C8-A12V-01Z-00-DX1_cells_fast.geojson

echo Conversion complete.
