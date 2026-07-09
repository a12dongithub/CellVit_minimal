@echo off
echo Running Batch Conversion in Docker...
docker run --rm ^
  --entrypoint /opt/conda/bin/python ^
  -v "%cd%\results":/results ^
  -v "%cd%\convert_tiles_batch.py":/convert_tiles_batch.py ^
  ikimhoerst/cellvit:beta ^
  /convert_tiles_batch.py
