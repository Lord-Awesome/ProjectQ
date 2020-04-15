//TODO: I got this off of stackoverflow. I don't know if we actually have thrust
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
#define MAT_DIM 2
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
    double real;
    double imag;
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

__global__ void one_qubit_kernel(complex* vec, int vec_size, int qubit_id, int elements_per_chunk) {

    //qid0 is smaller than qid
    int elements_per_thread = MAT_DIM; //1 quibit kernel
    int qid0 = qubit_id;
    int blocks_in_state_vector = ceil(vec_size / (float) (elements_per_thread * blockDim.x));
    int batch0_stride = (1 << (qid0 + 1));

    for(int global_block_id = blockIdx.x; global_block_id < blocks_in_state_vector; global_block_id += gridDim.x) {


        int element_id_base = 0;

		int batch0_id = (threadIdx.x + global_block_id * blockDim.x) / (batch0_stride/2);
		element_id_base += (threadIdx.x + global_block_id * blockDim.x) % (batch0_stride/2);
		element_id_base += batch0_id * batch0_stride;

        //iteration dependent

        complex result[MAT_DIM];
        for(int row = 0; row < MAT_DIM; row++) {
            result[row] = C(0.0f, 0.0f);
        }
        for(int i = 0; i < 2; i++) {
            int offset = (i * (1 << qid0));
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
            int column = i;
            for(int row = 0; row < MAT_DIM; row++) {
                result[row] = result[row] + (operator_matrix[row][column]*val);
            }
        }

        for(int i = 0; i < 2; i++) {
            int offset = (i * (1 << qid0));
            int element_id = element_id_base + offset;

            //store
            int row = i;
            if(element_id < vec_size) {
                vec[element_id] = result[row];
            }
        }


    }
}

//TODO: Header
template <class M>
void run_kernel(complex* vec, int vec_size, int qubit_id, M source_matrix) {
    cudaDeviceSynchronize();

    //Get smem size
    cudaDeviceProp deviceProp;
    int dev_id = 0;
    cudaGetDeviceProperties(&deviceProp, dev_id);
    int smem_size_in_bytes = (int) deviceProp.sharedMemPerBlock;
    int smem_size_in_elems = smem_size_in_bytes/(2*sizeof(double));

    int max_threads_per_block = (int) deviceProp.maxThreadsPerBlock;

    // calculate
    int num_loads_to_sm = 0;
    int num_large_qubits = (1 << (qubit_id + 1)) > max_threads_per_block;
    num_loads_to_sm += num_large_qubits;
    num_loads_to_sm = (1UL << num_loads_to_sm);

//A chunk can't be larger than shared memory because we need to hold it all at once
//A chunk can't be larger than the threads in a block because we need one thread to handle each element
//A chunk can't be larger than a batch by definition
    int chunk_size = max_threads_per_block;
    int chunk_size_in_bytes = chunk_size * sizeof(complex);
    dim3 blockDim(chunk_size);
int max_grid_size = deviceProp.maxGridSize[0];
    dim3 gridDim(std::min(max_grid_size, (int) ceil(vec_size/(float)chunk_size/(float)num_loads_to_sm)));

    //print some stats about the GPU
    std::cout << "smem_size_in_elems: " << smem_size_in_elems << std::endl;
    std::cout << "max_threads_per_block: " << max_threads_per_block << std::endl;;
    std::cout << "chunk size: " << chunk_size << std::endl;;
    std::cout << "max grid size: " << max_grid_size << std::endl;;

std::cout << "Vec size (num vectors is log2): " << vec_size << std::endl;
std::cout << "kth qubit: " << qubit_id << std::endl;

    std::cout << "block dim: " << blockDim.x << std::endl;
    std::cout << "grid dim: " << gridDim.x << std::endl;

    //memcpy and run the kernel
start = std::chrono::high_resolution_clock::now();

    complex *d_vec;
    cudaMalloc((void **) &d_vec, vec_size*sizeof(complex));
    cudaMemcpy(d_vec, vec, vec_size*sizeof(complex), cudaMemcpyHostToDevice);
    one_qubit_kernel<<<gridDim, blockDim, chunk_size_in_bytes>>>(d_vec, vec_size, qubit_id, chunk_size);
    cudaDeviceSynchronize();
    cudaMemcpy(vec, d_vec, vec_size*sizeof(complex), cudaMemcpyDeviceToHost);
    cudaFree(d_vec);

stop = std::chrono::high_resolution_clock::now();
}

int main(int argc, char **argv) {

if (argc != 3) {
std::cout << "Input args wrong. Needs total number qubits and kth qubit" << std::endl;
exit(1);
}


int kth_qubit = atoi(argv[2]);
    //int kth_qubit = 11;

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


//Read in source matrix
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

    run_kernel(state_vec.data(), state_vec_size, kth_qubit, source_matrix_vec.data());


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
