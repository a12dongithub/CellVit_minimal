@echo off
echo Removing existing container if any...
docker rm -f cellvit-inference 2>nul

echo Running Tile Sampling in Docker...
docker run --name cellvit-inference ^
  --rm ^
  --entrypoint /opt/conda/bin/python ^
  -v "%cd%\data":/data ^
  -v "%cd%\results":/results ^
  -v "%cd%\sample_tiles.py":/sample_tiles.py ^
  ikimhoerst/cellvit:beta ^
  /sample_tiles.py
