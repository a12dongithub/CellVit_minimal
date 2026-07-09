@echo off
echo Removing existing container if any...
docker rm -f cellvit-inference 2>nul

echo Inspecting JSON in Docker...
docker run --name cellvit-inference ^
  --rm ^
  --entrypoint /opt/conda/bin/python ^
  -v "%cd%\results":/results ^
  -v "%cd%\inspect_json.py":/inspect_json.py ^
  ikimhoerst/cellvit:beta ^
  /inspect_json.py ^
  /results/TCGA-C8-A12V-01Z-00-DX1_cells.json
