@echo off
echo Removing existing container if any...
docker rm -f cellvit-inference 2>nul

echo Creating Sample GeoJSON in Docker...
docker run --name cellvit-inference ^
  --rm ^
  --entrypoint /opt/conda/bin/python ^
  -v "%cd%\results":/results ^
  -v "%cd%\create_sample_geojson.py":/create_sample_geojson.py ^
  ikimhoerst/cellvit:beta ^
  /create_sample_geojson.py
