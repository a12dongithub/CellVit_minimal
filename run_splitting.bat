@echo off
echo Running Result Splitting in Docker...
docker run --rm ^
  --entrypoint /opt/conda/bin/python ^
  -v "%cd%\results":/results ^
  -v "%cd%\split_results.py":/split_results.py ^
  ikimhoerst/cellvit:beta ^
  /split_results.py ^
  --images /results/sampled_tiles_1024 ^
  --geojsons /results/sampled_tiles_1024_turbo_small_geojsons ^
  --output /results/final_512_dataset_benchmark
