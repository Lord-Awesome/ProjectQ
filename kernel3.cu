#include <cuComplex.h>
#include <complex>
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <chrono>

#define CUDACHECK(cmd) \
    cudaError_t error=cmd; \
    if(error != cudaSuccess) { \
        printf("%s in %s at line %d\n", cudaGetErrorString(error),__ \
        FILE__,__LINE__); \
        exit(EXIT_FAILURE); \
    }

#define FILENAME "state_vec.txt"
#define MAT_FILENAME "source_matrix.txt"
#define MAT_DIM 8
#define C(r, i) make_cuComplex(r, i)
typedef cuComplex complex;

//Overload complex number functions
__device__ __host__ complex operator*(complex a, complex b) {return cuCmulf(a,b);}
__device__ __host__ complex operator+(complex a, complex b) {return cuCaddf(a,b);}
__device__ __host__ complex operator/(complex a, complex b) {return cuCdivf(a,b);}
__device__ __host__ bool operator==(complex a, complex b) {
	return a == b;
}
__device__ __host__ bool operator!=(complex a, complex b) {
	return a != b;
}
__host__ std::ostream & operator << (std::ostream &out, const complex &c) {
	out << "(" << cuCrealf(c);
	out << ",";
	out << cuCimagf(c) << ")\n";
	return out;
}
__host__ std::istream & operator >> (std::istream &in, complex &c) {
    char _; //throw out variable
    float real;
    float imag;
    in >> _; //"("
    in >> real;
    in >> _; //","
    in >> imag;
    in >> _ >> _; //")\n";
    c = C(real, imag);
    return in;
}

__constant__ complex operator_matrix[MAT_DIM][MAT_DIM];

std::chrono::high_resolution_clock::time_point start, stop;

__global__ void three_qubit_kernel(complex* vec, int vec_size, int qid0, int qid1, int qid2, int elements_per_chunk) {
    //qid0 is smaller than qid1

    //Initialize shared memory
    extern __shared__ complex smem[];

    int elements_per_thread = MAT_DIM; //3 quibit kernel

    int blocks_in_state_vector = ceil(vec_size / (float) (elements_per_thread * blockDim.x));
	int batch0_stride = 2 * (1 << qid0);
	int batch1_stride = 2 * (1 << qid1);
	int batch2_stride = 2 * (1 << qid2);
    for(int global_block_id = blockIdx.x; global_block_id < blocks_in_state_vector; global_block_id += gridDim.x) {

        int element_id_base = 0;
		int global_thread_id = (threadIdx.x + global_block_id * blockDim.x);

		int batch1_depth = global_thread_id % (batch1_stride/4);
		int batch0_id = batch1_depth / (batch0_stride/2);

		element_id_base += global_thread_id % (batch0_stride/2);
		element_id_base += batch0_id * batch0_stride;

		int batch2_depth = global_thread_id % (batch2_stride/8);
		int batch1_id = batch2_depth / (batch1_stride/4);

		element_id_base += batch1_id * batch1_stride;

		int batch2_id = global_thread_id / (batch2_stride/8);

		element_id_base += batch2_id * batch2_stride;


        //iteration dependent

        complex result[MAT_DIM];
        for(int row = 0; row < MAT_DIM; row++) {
            result[row] = C(0.0f, 0.0f);
        }
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 2; j++) {
				for (int k = 0; k < 2; k++) {
					int offset = (i * (1 << qid2)) + (j * (1 << qid1)) + (k * (1 << qid0));
					int element_id = element_id_base + offset;

					//load
					complex val;
					if(element_id < vec_size) {
						val = vec[element_id];
					}
					else {
						val = C(0.0f,0.0f);
					}

					//compute
					int column = (4*i)+(2*j)+k;
					for(int row = 0; row < MAT_DIM; row++) {
						result[row] = result[row] + (operator_matrix[row][column]*val);
					}
				}//k
            }//j
        }//i

        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 2; j++) {
				for (int k = 0; k < 2; k++) {
					int offset = (i * (1 << qid2)) + (j * (1 << qid1)) + (k * (1 << qid0));
					int element_id = element_id_base + offset;

					//store
					int row = (4*i)+(2*j)+k;
					if(element_id < vec_size) {
						vec[element_id] = result[row];
						//vec[element_id] = C((float)element_id_base,(float)offset);
						//vec[element_id] = C((float)global_thread_id, (float)element_id_base);
					}
				}//k
            }//j
        }//i

    }
}

//TODO: Header
template <class M>
void run_kernel(complex* vec, int vec_size, int quid0, int quid1, int quid2, M source_matrix) {
    cudaDeviceSynchronize();

    //Get smem size
    cudaDeviceProp deviceProp;
    int dev_id = 0;
    cudaGetDeviceProperties(&deviceProp, dev_id);
    int smem_size_in_bytes = (int) deviceProp.sharedMemPerBlock;
    int smem_size_in_elems = smem_size_in_bytes/(2*sizeof(double));

    int max_threads_per_block = (int) deviceProp.maxThreadsPerBlock;

	//A chunk can't be larger than shared memory because we need to hold it all at once
	//A chunk can't be larger than the threads in a block because we need one thread to handle each element
	//A chunk can't be larger than a batch by definition
    //int chunk_size = std::min(std::min(smem_size_in_elems, max_threads_per_block),(int) batch_size);
	int chunk_size = max_threads_per_block;
    int chunk_size_in_bytes = chunk_size * sizeof(complex);
    dim3 blockDim(chunk_size);
	int max_grid_size = deviceProp.maxGridSize[0];
    dim3 gridDim(std::min(max_grid_size, (int) ceil(vec_size/(float)chunk_size)));

    //print some stats about the GPU
    std::cout << "smem_size_in_elems: " << smem_size_in_elems << std::endl;
    std::cout << "max_threads_per_block: " << max_threads_per_block << std::endl;;
    std::cout << "chunk size: " << chunk_size << std::endl;;
    std::cout << "max grid size: " << max_grid_size << std::endl;;

	std::cout << "Vec size (num vectors is log2): " << vec_size << std::endl;
	std::cout << "quid0: " << quid0 << std::endl;
	std::cout << "quid1: " << quid1 << std::endl;
	std::cout << "quid2: " << quid2 << std::endl;

    std::cout << "block dim: " << blockDim.x << std::endl;
    std::cout << "grid dim: " << gridDim.x << std::endl;

    //memcpy and run the kernel
	start = std::chrono::high_resolution_clock::now();

    complex *d_vec;
    cudaMalloc((void **) &d_vec, vec_size*sizeof(complex));
    cudaError_t cpy_error = cudaMemcpy(d_vec, vec, vec_size*sizeof(complex), cudaMemcpyHostToDevice);
	//std::cout << "Copying to device error is: " << cpy_error << std::endl;
    three_qubit_kernel<<<gridDim, blockDim, chunk_size_in_bytes>>>(d_vec, vec_size, quid0, quid1, quid2, chunk_size);
	//cudaError_t kernel_error = cudaGetLastError();
	//std::cout << "Kernel error is: " << kernel_error << std::endl;
    cudaDeviceSynchronize();
    cpy_error = cudaMemcpy(vec, d_vec, vec_size*sizeof(complex), cudaMemcpyDeviceToHost);
	//std::cout << "Copying to host error is: " << cpy_error << std::endl;
    cudaFree(d_vec);

	stop = std::chrono::high_resolution_clock::now();
}

int main(int argc, char **argv) {

	if (argc != 5) {
		std::cout << "Input args wrong. Needs exactly four input args" << std::endl;
		exit(1);
	}


	int quid0 = atoi(argv[2]);
	int quid1 = atoi(argv[3]);
	int quid2 = atoi(argv[4]);

    //Read state vector
    std::vector<complex> state_vec;
    std::ifstream fin;
    complex temp;
	std::complex<float> std_complex_temp;


    fin.open(FILENAME);
    while(fin >> std_complex_temp) {
		temp = C(std_complex_temp.real(), std_complex_temp.imag());
        state_vec.push_back(temp);
    }
	if (fin.rdstate() == std::ios_base::failbit) {
		std::cout << "Ifstream failed with failbit" << std::endl;
	}
	else if (fin.rdstate() == std::ios_base::eofbit) {
		std::cout << "Ifstream failed with eofbit" << std::endl;
	}
	else if (fin.rdstate() == std::ios_base::badbit) {
		std::cout << "Ifstream failed with badbit" << std::endl;
	}
	std::cout << "Vector size: " << state_vec.size() << std::endl;
    fin.close();

	/*
    for (unsigned long i = 0; i < 1 << atoi(argv[1]); i++){ 
        //Note: normalization ignored for now
        float real = ((float) rand() / (float) (RAND_MAX));
        float imag = ((float) rand() / (float) (RAND_MAX));
        complex val = C(real, imag);
		state_vec.push_back(val);
    }
	*/

    unsigned long state_vec_size = state_vec.size();


    std::vector<complex> source_matrix_vec;
	std::cout << "here is the source matrix: " << std::endl;
    fin.open(MAT_FILENAME);
    while(fin >> std_complex_temp) {
		temp = C(std_complex_temp.real(), std_complex_temp.imag());
        source_matrix_vec.push_back(temp);
		std::cout << temp << std::endl;
    }
	if (fin.rdstate() == std::ios_base::failbit) {
		std::cout << "Ifstream failed with failbit" << std::endl;
	}
	else if (fin.rdstate() == std::ios_base::eofbit) {
		std::cout << "Ifstream failed with eofbit" << std::endl;
	}
	else if (fin.rdstate() == std::ios_base::badbit) {
		std::cout << "Ifstream failed with badbit" << std::endl;
	}
    fin.close();

    //Apply gate
    //Fill operator matrix in const mem
    cudaMemcpyToSymbol(operator_matrix, source_matrix_vec.data(), MAT_DIM * MAT_DIM * sizeof(complex), 0, cudaMemcpyHostToDevice);

    run_kernel(state_vec.data(), state_vec_size, quid0, quid1, quid2, source_matrix_vec.data());


	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
	std::cout << "GPU kernel execution time: " << duration.count() << std::endl;

	std::ofstream f_time;
	f_time.open("time_comparison.txt", std::ios_base::app);
	f_time << "GPU time: " << duration.count() << "\n";
	f_time.close();
 
    std::ofstream f;
    f.open("output.txt");
    for (unsigned long i = 0; i < state_vec_size; ++i) {
        complex val = state_vec[i];
        f << val;	
    }
    f.close();
    
    //debug
    std::cout << "size: " << state_vec.size() << std::endl;
    std::cout << state_vec.back() << std::endl;


    return 0;
}
