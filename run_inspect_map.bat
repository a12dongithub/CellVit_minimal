@echo off
echo Inspecting type_map...
docker run --rm ^
  --entrypoint /opt/conda/bin/python ^
  -v "%cd%\results":/results ^
  -v "%cd%\inspect_type_map.py":/inspect_type_map.py ^
  ikimhoerst/cellvit:beta ^
  /inspect_type_map.py
