#include "kernels.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>

typedef std::complex<double> complex;
#define C(r, i) complex(r, i)
#define VEC_FILENAME "state_vec.txt"
#define MAT_FILENAME "source_matrix.txt"

int main(int argc, char **argv) {
	if (argc < 3) {
		std::cout << "Input args wrong. First arg is number of qubits to operate on. Then next args are those qubit IDs" << std::endl;
		exit(1);
	}

    //Read in state vec
    std::vector<complex> state_vec;
    std::ifstream fin;
    fin.open(VEC_FILENAME);
    complex temp;
    while(fin >> temp) {
        state_vec.push_back(temp);
    }
    fin.close();

	//Read in source matrix
	int mat_dim = 1<<atoi(argv[1]);
	complex source_matrix[32][32];
	fin.open(MAT_FILENAME);
	if (!fin.is_open()) {
		std::cout << "Source matrix does not exist!" << std::endl;
		exit(1);
	}
	for (int i = 0; i < mat_dim; i++) {
		for (int j = 0; j < mat_dim; j++) {
			fin >> temp;
			source_matrix[i][j] = temp;
		}
	}
	fin.close();

	auto start = std::chrono::high_resolution_clock::now();

    //Apply NOT gate
	switch (atoi(argv[1])) {
		case 1:
			//1 qubit
			kernel(state_vec, atoi(argv[2]), source_matrix, 0);
			break;
		case 2:
			//2 qubits
			kernel(state_vec, atoi(argv[2]), atoi(argv[3]), source_matrix, 0);
			break;
		case 3:
			//3 qubits
			kernel(state_vec, atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), source_matrix, 0);
			break;
		case 4:
			//4 qubits
			kernel(state_vec, atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), source_matrix, 0);
			break;
		case 5:
			//5 qubits
			kernel(state_vec, atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), source_matrix, 0);
			break;
	}
 
	auto stop = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
	std::cout << "Intrin baseline time: " << duration.count() << std::endl;

	std::ofstream f_time;
	f_time.open("time_comparison.txt", std::ios_base::app);
	f_time << "Intrin baseline time: " << duration.count() << "\n";
	f_time.close();

    std::ofstream fout;
    fout.open("output_truth.txt");
    for (size_t i = 0; i < state_vec.size(); ++i) {
	complex val = state_vec[i];
	fout << val << "\n";	
    }
    fout.close();
    return 0;
}
