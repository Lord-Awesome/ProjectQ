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
if [$1 -eq 1]
then
	./kernel1.o $2
fi
if [$1 -eq 2]
then
	./kernel2.o $2 $3
fi
if [$1 -eq 3]
then
	./kernel3.o $2 $3 $4
fi
if [$1 -eq 4]
then
	./kernel4.o $2 $3 $4 $5
fi
if [$1 -eq 5]
then
	./kernel5.o $2 $3 $4 $5 $6
fi
