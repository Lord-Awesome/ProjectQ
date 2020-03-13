#include<stdlib.h> //rand,srand
#include<vector>
#include<fstream>
#include<complex>

#define SEED 1234
#define NUM_QUBITS 14
#define FILENAME "state_vec.txt"

typedef std::complex<double> complex;
#define C(r, i) complex(r, i)

int main() {
    srand(SEED);
    std::ofstream f;
    f.open(FILENAME);

    unsigned long state_vec_size = 1UL << NUM_QUBITS;
    std::vector<complex> state_vec(state_vec_size, C(0.0f, 0.0f));
    for (unsigned long i = 0; i < state_vec_size; i++){ 
        //Note: normalization ignored for now
        float real = ((double) rand() / (RAND_MAX));
        float imag = ((double) rand() / (RAND_MAX));
        complex val = C(real, imag);
        state_vec[i] = val;
        f << val << "\n";
    }
    f.close();
    return 0;
}