#!/bin/bash
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -l walltime=24:00:00
#PBS -q <queue_name>
#PBS -N pytorch_check
#PBS -j oe

# Change to your working directory
cd $PBS_O_WORKDIR

# Activate the Conda environment
source activate base

# Load necessary modules (if needed)
# module load cuda cudnn (Uncomment if not loaded automatically or installed separately)

# Run the Python script
python check_pytorch.py
