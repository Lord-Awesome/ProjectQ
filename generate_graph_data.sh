NUM_QUBIT_IDS=2
MIN_NUM_QUBITS=16
MAX_NUM_QUBITS=20

#echo "\n\n---------- Removing time_comparison.txt ----------\n\n"
rm time_comparison.txt

#echo "\n\n---------- Compiling with G++----------\n\n"
#g++ --std=c++11 gen_state_vec.cpp -o gen_state_vec.o || exit 1
#g++ --std=c++11 gen_source_matrix.cpp -o gen_source_matrix.o || exit 1
#g++ -g --std=c++11 projectq_kernel_nointrin_runner.cpp -o projectq_kernel_nointrin_runner.o -I./projectq/backends/_sim/_cppkernels/nointrin/ || exit 1
#g++ -g --std=c++11 -mavx projectq_kernel_intrin_runner.cpp -o projectq_kernel_intrin_runner.o -I./projectq/backends/_sim/_cppkernels/intrin/ || exit 1
#echo "\n\n---------- Compiling with NVCC----------\n\n"
#nvcc --std=c++11 kernel1.cu -o kernel1.o || exit 1
#nvcc --std=c++11 kernel2.cu -o kernel2.o || exit 1
#nvcc --std=c++11 kernel3.cu -o kernel3.o || exit 1
#nvcc --std=c++11 kernel4.cu -o kernel4.o || exit 1
#nvcc --std=c++11 kernel5.cu -o kernel5.o || exit 1

for ((NUM_QUBITS=$MIN_NUM_QUBITS; NUM_QUBITS<=$MAX_NUM_QUBITS; NUM_QUBITS++));
do
	#for ((QUBIT0=0; QUBIT0<=$NUM_QUBITS-2; QUBIT0++));
	#do
	TEMP1=$(($NUM_QUBITS - 2))
	#QUBIT0=$(shuf -i 0-$TEMP1 -n 1)
	QUBIT0=($(shuf -e $(seq 0 $TEMP1) -n 1))
		#for ((QUBIT1=$QUBIT0+1; QUBIT1<=$NUM_QUBITS-1; QUBIT1++));
		TEMP2=$(($QUBIT0 + 1))
		TEMP3=$(($NUM_QUBITS - 1))
		#QUBIT1=$(shuf -i $TEMP2-$TEMP3 -n 1)
		QUBIT1=($(shuf -e $(seq $TEMP2 $TEMP3) -n 1))
		#do
			echo "NUM_QUBITS: $NUM_QUBITS QUBIT1: $QUBIT1 QUBIT0: $QUBIT0" >> time_comparison.txt
			echo "NUM_QUBITS: $NUM_QUBITS QUBIT1: $QUBIT1 QUBIT0: $QUBIT0"

			#echo "\n\n---------- Generating truth ----------\n\n"
			./gen_state_vec.o $NUM_QUBITS || exit 1
#~/Python-3.8.1/python3.8 source_generate.py $NUM_QUBIT_IDS
			./gen_source_matrix.o $NUM_QUBIT_IDS || exit 1

			./projectq_kernel_nointrin_runner.o $NUM_QUBIT_IDS $QUBIT1 $QUBIT0 || exit 1
			./projectq_kernel_intrin_runner.o $NUM_QUBIT_IDS $QUBIT1 $QUBIT0 || exit 1

			#echo "\n\n---------- Running job on GPU ----------\n\n"
			sbatch --wait run_on_gpu.sh $NUM_QUBIT_IDS $QUBIT0 $QUBIT1

			#echo "\n\n---------- Job Done ----------\n\n"

			#echo "\n\n---------- Copying log file ----------\n\n"
			cp ~/570_job* slurm_job_output/
			rm ~/570_job*

			#echo "\n\n---------- Comparing output ----------\n\n"
			~/Python-3.8.1/python3.8 compare_outputs.py
		#done
	#done
done
