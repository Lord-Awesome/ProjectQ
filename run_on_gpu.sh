#!/bin/bash
#SBATCH --job-name=570_job
#SBATCH --mail-user=<uniqname>@umich.edu
#SBATCH --mail-type=END
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=2500m
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
# $1 is the num qubits
# $2 is the kth qubit
# $3, $4, etc. are the qubitids
if [ $2 -eq 1 ]
then
	./kernel1.o $1 $3
fi
if [ $2 -eq 2 ]
then
	./kernel2.o $1 $3 $4
fi
if [ $2 -eq 3 ]
then
	# rm kernel3_analysis.prof
	# nvprof --analysis-metrics --track-memory-allocations on -o kernel3_analysis.prof ./kernel3.o $1 $3 $4 $5
	./kernel3.o $1 $3 $4 $5
fi
if [ $2 -eq 4 ]
then
	./kernel4.o $1 $3 $4 $5 $6
fi
if [ $2 -eq 5 ]
then
	#rm kernel5_analysis.prof
	#nvprof --analysis-metrics --track-memory-allocations on -o kernel5_analysis.prof ./kernel5.o $1 $3 $4 $5 $6 $7
	./kernel5.o $1 $3 $4 $5 $6 $7
fi
