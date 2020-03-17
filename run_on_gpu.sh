#!/bin/bash
#SBATCH --job-name=570_job
#SBATCH --mail-user=<uniqname>@umich.edu
#SBATCH --mail-type=END
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1000m 
#SBATCH --time=10:00
#SBATCH --account=eecs570w20_class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/home/%u/%x-%j.srun.log

# Load CUDA
module load cuda
#nvcc --version

# Compile
#cd ~/src
#nvcc code.cu -o binary

# Run
# $1 is the kth qubit
./kernel1.o $1
