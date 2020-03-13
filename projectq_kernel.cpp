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
#include "kernels.hpp"

typedef std::complex<double> complex;
#define C(r, i) complex(r, i)
#define FILENAME "state_vec.txt"

template <class V, class M>
inline void kernel_core(V &psi, std::size_t I, std::size_t d0, M const& m)
{
    std::complex<double> v[2];
    v[0] = psi[I];
    v[1] = psi[I + d0];

    psi[I] = (add(mul(v[0], m[0][0]), mul(v[1], m[0][1])));
    psi[I + d0] = (add(mul(v[0], m[1][0]), mul(v[1], m[1][1])));

}

// bit indices id[.] are given from high to low (e.g. control first for CNOT)
template <class V, class M>
void kernel(V &psi, unsigned id0, M const& m, std::size_t ctrlmask)
{
    std::size_t n = psi.size();
    std::size_t d0 = 1UL << id0;
    std::size_t dsorted[] = {d0 };
    std::sort(dsorted, dsorted + 1, std::greater<std::size_t>());

    if (ctrlmask == 0){
        #pragma omp for collapse(LOOP_COLLAPSE1) schedule(static)
        for (std::size_t i0 = 0; i0 < n; i0 += 2 * dsorted[0]){
            for (std::size_t i1 = 0; i1 < dsorted[0]; ++i1){
                kernel_core(psi, i0 + i1, d0, m);
            }
        }
    }
    else{
        #pragma omp for collapse(LOOP_COLLAPSE1) schedule(static)
        for (std::size_t i0 = 0; i0 < n; i0 += 2 * dsorted[0]){
            for (std::size_t i1 = 0; i1 < dsorted[0]; ++i1){
                if (((i0 + i1)&ctrlmask) == ctrlmask)
                    kernel_core(psi, i0 + i1, d0, m);
            }
        }
    }
}

int main() {
    int kth_qubit = 11;

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

    //Apply NOT gate
    kernel(state_vec, kth_qubit, source_matrix, 0);
    
    std::ofstream fout;
    fout.open("output_truth.txt");
    for (size_t i = 0; i < state_vec.size(); ++i) {
	complex val = state_vec[i];
	fout << val << "\n";	
    }
    fout.close();
    return 0;
}
