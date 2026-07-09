@echo off
echo Listing /results/sampled_tiles...
docker run --rm ^
  --entrypoint /bin/bash ^
  -v "%cd%\results":/results ^
  ikimhoerst/cellvit:beta ^
  -c "ls -lh /results/sampled_tiles | head -n 5"

echo Listing /results/sampled_tiles_inference...
docker run --rm ^
  --entrypoint /bin/bash ^
  -v "%cd%\results":/results ^
  ikimhoerst/cellvit:beta ^
  -c "ls -lh /results/sampled_tiles_inference"
