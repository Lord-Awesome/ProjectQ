#include<stdlib.h> //rand,srand
#include<vector>
#include<fstream>
#include <iostream>
#include<complex>

#define SEED 1234
#define FILENAME "state_vec.txt"

typedef std::complex<float> complex;
#define C(r, i) complex(r, i)

int main(int argc, char **argv) {
	if (argc != 2) {
		std::cout << "Input args wrong. Needs exactly one input arg which is the total number of qubits" << std::endl;
		exit(1);
	}
	
	int NUM_QUBITS = atoi(argv[1]);

    srand(SEED);
    std::ofstream f;
    f.open(FILENAME);

    unsigned long state_vec_size = 1UL << NUM_QUBITS;
    std::vector<complex> state_vec(state_vec_size, C(0.0f, 0.0f));
    for (unsigned long i = 0; i < state_vec_size; i++){ 
        //Note: normalization ignored for now
        float real = ((float) rand() / (RAND_MAX));
        float imag = ((float) rand() / (RAND_MAX));
        complex val = C(real, imag);
        state_vec[i] = val;
        f << val << "\n";
    }
    f.close();
    return 0;
}
