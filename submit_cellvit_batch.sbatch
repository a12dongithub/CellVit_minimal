#!/bin/bash
#SBATCH --job-name=cellvit_batch
#SBATCH --partition=talon-gpu32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:6
#SBATCH --time=90:00:00
#SBATCH --output=cellvit_batch_%j.log
#SBATCH --error=cellvit_batch_%j.err
#SBATCH --mail-type=END,FAIL

echo "Starting CellViT batch job on $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

# Change to the directory where the job was submitted from
cd $SLURM_SUBMIT_DIR

# Run the pipeline script
bash ./run_cloud_batch.sh --data_dir ../data/ --model models/CellViT-SAM-H-x40-AMP-001.pth --num_gpus 6 --batch_size 8

echo "Job completed."
