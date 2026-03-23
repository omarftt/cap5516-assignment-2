#!/bin/bash
#SBATCH --job-name=brats_seg
#SBATCH --cpus-per-task=6
#SBATCH --qos=short
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --constraint="ampere"
#SBATCH --time=08:00:00

# Generate random 4-digit ID for this run
RANDOM_ID=$(printf "%04d" $((RANDOM % 10000)))

# Create logs folder if it doesn't exist
mkdir -p logs

# Redirect output and error to log files
exec > logs/log_${RANDOM_ID}.txt 2> logs/error_${RANDOM_ID}.txt

# Print job info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Log ID: $RANDOM_ID"
echo "================================"

# Activate virtual environment
source venv/bin/activate

# Run 5-fold cross-validation
python -u main.py --mode crossval

# Print completion info
echo "================================"
echo "Job completed at: $(date)"