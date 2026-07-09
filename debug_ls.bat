@echo off
echo Listing files in /results...
docker run --rm ^
  --entrypoint /bin/bash ^
  -v "%cd%\results":/results ^
  ikimhoerst/cellvit:beta ^
  -c "ls -lh /results"
