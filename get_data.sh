NUM_ITER=5

for ((ITER=0; ITER<$NUM_ITER; ITER++));
do
	START_TIME=$SECONDS

	echo $'\n\n---------- Running graph_effect_of_operator_size.sh ----------\n\n'
	./graph_effect_of_operator_size.sh
	cp data/graph_data_operator_size.txt data/iterations/graph_data_operator_size_iter_${ITER}.txt
	echo $'\n\n---------- Running graph_effect_of_qubit_ids.sh ----------\n\n'
	./graph_effect_of_qubit_ids.sh
	cp data/graph_data_qubitid_magnitude.txt data/iterations/graph_data_qubitid_magnitude_iter_${ITER}.txt
	echo $'\n\n---------- Running graph_effect_of_qubit_spacing.sh ----------\n\n'
	./graph_effect_of_qubit_spacing.sh
	cp data/graph_data_qubitid_spacing.txt data/iterations/graph_data_qubitid_spacing_iter_${ITER}.txt
	echo $'\n\n---------- Running graph_effect_of_state_vec_size.sh ----------\n\n'
	./graph_effect_of_state_vec_size.sh
	cp data/graph_data_state_vec_size.txt data/iterations/graph_data_state_vec_size_iter_${ITER}.txt

	DURATION=$(($SECONDS - $START_TIME))
	echo "Iteration $(($ITER)) is done with time $(($DURATION))"
done
