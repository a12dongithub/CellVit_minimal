@echo off
echo Reading pathopatch/patch_extraction/dataset.py...
docker run --rm ^
  --entrypoint /bin/bash ^
  ikimhoerst/cellvit:beta ^
  -c "cat /opt/conda/lib/python3.11/site-packages/pathopatch/patch_extraction/dataset.py"
