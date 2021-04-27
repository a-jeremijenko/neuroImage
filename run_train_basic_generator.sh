#!/bin/bash
#SBATCH -p short
#SBATCH -t 30
#SBATCH -n 2
#SBATCH --mem=12000
#SBATCH --job-name basic_generator
#SBATCH --output basic_generator-%j.out

# Set up the environment
module load miniconda
conda activate mybrainiak


srun python ./train_basic_generator.py

