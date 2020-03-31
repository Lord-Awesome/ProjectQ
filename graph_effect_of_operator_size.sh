NUM_QUBITS=22

MIN_NUM_QUBIT_IDS=1
MAX_NUM_QUBIT_IDS=4

#echo "\n\n---------- Removing time_comparison.txt ----------\n\n"
rm time_comparison.txt

#echo "\n\n---------- Compiling with G++----------\n\n"
g++ --std=c++11 gen_state_vec.cpp -o gen_state_vec.o || exit 1
g++ --std=c++11 gen_source_matrix.cpp -o gen_source_matrix.o || exit 1
g++ -g --std=c++11 projectq_kernel_nointrin_runner.cpp -o projectq_kernel_nointrin_runner.o -I./projectq/backends/_sim/_cppkernels/nointrin/ || exit 1
g++ -g --std=c++11 -mavx projectq_kernel_intrin_runner.cpp -o projectq_kernel_intrin_runner.o -I./projectq/backends/_sim/_cppkernels/intrin/ || exit 1

#echo "\n\n---------- Compiling with NVCC----------\n\n"
nvcc --std=c++11 kernel1.cu -o kernel1.o || exit 1
nvcc --std=c++11 kernel2.cu -o kernel2.o || exit 1
nvcc --std=c++11 kernel3.cu -o kernel3.o || exit 1
nvcc --std=c++11 kernel4.cu -o kernel4.o || exit 1
nvcc --std=c++11 kernel5.cu -o kernel5.o || exit 1

for ((NUM_QUBIT_IDS=$MIN_NUM_QUBIT_IDS; NUM_QUBIT_IDS<=$MAX_NUM_QUBIT_IDS; NUM_QUBIT_IDS++));
do
	QUBIT4=$(($NUM_QUBITS - 1))
	QUBIT3=$(($NUM_QUBITS - 2))
	QUBIT2=$(($NUM_QUBITS - 3))
	QUBIT1=$(($NUM_QUBITS - 4))
	QUBIT0=$(($NUM_QUBITS - 5))

	if [ $NUM_QUBIT_IDS -eq 1 ]
	then
		echo "NUM_QUBITS: $NUM_QUBITS QUBIT0: $QUBIT0" >> time_comparison.txt
		echo "NUM_QUBITS: $NUM_QUBITS QUBIT0: $QUBIT0"
	fi
	if [ $NUM_QUBIT_IDS -eq 2 ]
	then
		echo "NUM_QUBITS: $NUM_QUBITS QUBIT1: $QUBIT1 QUBIT0: $QUBIT0" >> time_comparison.txt
		echo "NUM_QUBITS: $NUM_QUBITS QUBIT1: $QUBIT1 QUBIT0: $QUBIT0"
	fi
	if [ $NUM_QUBIT_IDS -eq 3 ]
	then
		echo "NUM_QUBITS: $NUM_QUBITS QUBIT2: $QUBIT2 QUBIT1: $QUBIT1 QUBIT0: $QUBIT0" >> time_comparison.txt
		echo "NUM_QUBITS: $NUM_QUBITS QUBIT2: $QUBIT2 QUBIT1: $QUBIT1 QUBIT0: $QUBIT0"
	fi
	if [ $NUM_QUBIT_IDS -eq 4 ]
	then
		echo "NUM_QUBITS: $NUM_QUBITS QUBIT3: $QUBIT3 QUBIT2: $QUBIT2 QUBIT1: $QUBIT1 QUBIT0: $QUBIT0" >> time_comparison.txt
		echo "NUM_QUBITS: $NUM_QUBITS QUBIT3: $QUBIT3 QUBIT2: $QUBIT2 QUBIT1: $QUBIT1 QUBIT0: $QUBIT0"
	fi
	if [ $NUM_QUBIT_IDS -eq 5 ]
	then
		echo "NUM_QUBITS: $NUM_QUBITS QUBIT4: $QUBIT4 QUBIT3: $QUBIT3 QUBIT2: $QUBIT2 QUBIT1: $QUBIT1 QUBIT0: $QUBIT0" >> time_comparison.txt
		echo "NUM_QUBITS: $NUM_QUBITS QUBIT4: $QUBIT4 QUBIT3: $QUBIT3 QUBIT2: $QUBIT2 QUBIT1: $QUBIT1 QUBIT0: $QUBIT0"
	fi



	#echo "\n\n---------- Generating truth ----------\n\n"
	./gen_state_vec.o $NUM_QUBITS || exit 1
#~/Python-3.8.1/python3.8 source_generate.py $NUM_QUBIT_IDS
	./gen_source_matrix.o $NUM_QUBIT_IDS || exit 1

	if [ $NUM_QUBIT_IDS -eq 1 ]
	then
		./projectq_kernel_nointrin_runner.o $NUM_QUBIT_IDS $QUBIT0 || exit 1
		./projectq_kernel_intrin_runner.o $NUM_QUBIT_IDS $QUBIT0 || exit 1
	fi
	if [ $NUM_QUBIT_IDS -eq 2 ]
	then
		./projectq_kernel_nointrin_runner.o $NUM_QUBIT_IDS $QUBIT1 $QUBIT0 || exit 1
		./projectq_kernel_intrin_runner.o $NUM_QUBIT_IDS $QUBIT1 $QUBIT0 || exit 1
	fi
	if [ $NUM_QUBIT_IDS -eq 3 ]
	then
		./projectq_kernel_nointrin_runner.o $NUM_QUBIT_IDS $QUBIT2 $QUBIT1 $QUBIT0 || exit 1
		./projectq_kernel_intrin_runner.o $NUM_QUBIT_IDS $QUBIT2 $QUBIT1 $QUBIT0 || exit 1
	fi
	if [ $NUM_QUBIT_IDS -eq 4 ]
	then
		./projectq_kernel_nointrin_runner.o $NUM_QUBIT_IDS $QUBIT3 $QUBIT2 $QUBIT1 $QUBIT0 || exit 1
		./projectq_kernel_intrin_runner.o $NUM_QUBIT_IDS $QUBIT3 $QUBIT2 $QUBIT1 $QUBIT0 || exit 1
	fi
	if [ $NUM_QUBIT_IDS -eq 5 ]
	then
		./projectq_kernel_nointrin_runner.o $NUM_QUBIT_IDS $QUBIT4 $QUBIT3 $QUBIT2 $QUBIT1 $QUBIT0 || exit 1
		./projectq_kernel_intrin_runner.o $NUM_QUBIT_IDS $QUBIT4 $QUBIT3 $QUBIT2 $QUBIT1 $QUBIT0 || exit 1
	fi

	#echo "\n\n---------- Running job on GPU ----------\n\n"
#--wait allows bash to wait for it to be done
	if [ $NUM_QUBIT_IDS -eq 1 ]
	then
		sbatch --wait run_on_gpu.sh $NUM_QUBIT_IDS $QUBIT0
	fi
	if [ $NUM_QUBIT_IDS -eq 2 ]
	then
		sbatch --wait run_on_gpu.sh $NUM_QUBIT_IDS $QUBIT0 $QUBIT1
	fi
	if [ $NUM_QUBIT_IDS -eq 3 ]
	then
		sbatch --wait run_on_gpu.sh $NUM_QUBIT_IDS $QUBIT0 $QUBIT1 $QUBIT2
	fi
	if [ $NUM_QUBIT_IDS -eq 4 ]
	then
		sbatch --wait run_on_gpu.sh $NUM_QUBIT_IDS $QUBIT0 $QUBIT1 $QUBIT2 $QUBIT3
	fi
	if [ $NUM_QUBIT_IDS -eq 5 ]
	then
		sbatch --wait run_on_gpu.sh $NUM_QUBIT_IDS $QUBIT0 $QUBIT1 $QUBIT2 $QUBIT3 $QUBIT4
	fi

#bash command
	wait

	#echo "\n\n---------- Job Done ----------\n\n"

	#echo "\n\n---------- Copying log file ----------\n\n"
	cp ~/570_job* slurm_job_output/
	rm ~/570_job*

	#echo "\n\n---------- Comparing output ----------\n\n"
	~/Python-3.8.1/python3.8 compare_outputs.py

	cp time_comparison.txt graph_data_operator_size.txt
done
