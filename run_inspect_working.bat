@echo off
echo Removing existing container if any...
docker rm -f cellvit-inference 2>nul

echo Inspecting Working GeoJSON in Docker...
docker run --name cellvit-inference ^
  --rm ^
  --entrypoint /opt/conda/bin/python ^
  -v "%cd%\results":/results ^
  -v "%cd%\inspect_working_geojson.py":/inspect_working_geojson.py ^
  ikimhoerst/cellvit:beta ^
  /inspect_working_geojson.py
