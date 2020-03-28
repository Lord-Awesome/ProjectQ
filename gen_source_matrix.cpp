#include<stdlib.h> //rand,srand
#include<vector>
#include<fstream>
#include <iostream>
#include<complex>

#define SEED 1234
#define FILENAME "source_matrix.txt"

typedef std::complex<float> complex;
#define C(r, i) complex(r, i)

int main(int argc, char **argv) {
	if (argc != 2) {
		std::cout << "Input args wrong. Needs exactly one input arg which is the number of qubit ids" << std::endl;
		exit(1);
	}
	
	int MATDIM = 1<<atoi(argv[1]);

    srand(SEED);
    std::ofstream f;
    f.open(FILENAME);

    for (unsigned long i = 0; i < MATDIM; i++){
		for (unsigned long j = 0; j < MATDIM; j++){
			float real = ((float) rand() / (RAND_MAX));
			float imag = ((float) rand() / (RAND_MAX));
			if (i == j) {
				real = 1.0f;
				imag = 0.0f;
			}
			else {
				real = 0.0f;
				imag = 0.0f;
			}
			complex val = C(real, imag);
			f << val << "\n";
		}
    }
    f.close();
    return 0;
}
