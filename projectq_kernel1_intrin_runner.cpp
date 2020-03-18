// Copyright 2017 ProjectQ-Framework (www.projectq.ch)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <vector>
#include <fstream>
#include <complex>
#include <algorithm>
#include "intrin_kernels.hpp"
#include <chrono>

typedef std::complex<double> complex;
#define C(r, i) complex(r, i)
#define FILENAME "state_vec.txt"

int main(int argc, char **argv) {
	if (argc != 2) {
		std::cout << "Input args wrong. Needs exactly one input arg which is the kth qubit" << std::endl;
		exit(1);
	}

	int kth_qubit = atoi(argv[1]);
    //int kth_qubit = 11;

    //Read in state vec
    std::vector<complex> state_vec;
    std::ifstream fin;
    fin.open(FILENAME);
    complex temp;
    while(fin >> temp) {
        state_vec.push_back(temp);
    }
    fin.close();

    complex source_matrix[2][2];
    source_matrix[0][0] = C(0.0f, 0.0f);
    source_matrix[0][1] = C(1.0f, 0.0f);
    source_matrix[1][0] = C(1.0f, 0.0f);
    source_matrix[1][1] = C(0.0f, 0.0f);

	auto start = std::chrono::high_resolution_clock::now();

    //Apply NOT gate
	//From  projectq_kernel1_intrin.cpp
    kernel(state_vec, kth_qubit, source_matrix, 0);
    
	auto stop = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
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
