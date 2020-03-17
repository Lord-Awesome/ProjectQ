NUM_QUBITS=20
KTH_QUBIT=16

echo "\n\n---------- Compiling ----------\n\n"
g++ --std=c++11 gen_state_vec.cpp -o gen_state_vec.o || exit 1
g++ --std=c++11 projectq_kernel.cpp kernels.hpp -o projectq_kernel.o || exit 1
nvcc --std=c++11 kernel1.cu -o kernel1.o || exit 1

echo "\n\n---------- Generating truth ----------\n\n"
./gen_state_vec.o $NUM_QUBITS || exit 1
./projectq_kernel.o $KTH_QUBIT || exit 1

echo "\n\n---------- Running job on GPU ----------\n\n"
#--wait allows bash to wait for it to be done
sbatch --wait run_on_gpu.sh $KTH_QUBIT

#bash command
wait

echo "\n\n---------- Job Done ----------\n\n"

echo "\n\n---------- Copying log file ----------\n\n"
cp ~/570_job* slurm_job_output/
rm ~/570_job*

vimdiff output.txt output_truth.txt





