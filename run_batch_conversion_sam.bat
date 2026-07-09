@echo off
echo Running Batch Conversion (SAM) in Docker...
docker run --rm ^
  --entrypoint /opt/conda/bin/python ^
  -v "%cd%\results":/results ^
  -v "%cd%\convert_tiles_batch_sam.py":/convert_tiles_batch_sam.py ^
  ikimhoerst/cellvit:beta ^
  /convert_tiles_batch_sam.py
