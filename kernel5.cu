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
#define MAT_DIM 32
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

__global__ void five_qubit_kernel(complex* vec, int vec_size, int qid0, int qid1, int qid2, int qid3, int qid4, int elements_per_chunk) {
    //qid0 is smaller than qid1

    //Initialize shared memory
    __shared__ int smem[20];

    int blocks_in_state_vector = ceil(vec_size / (float) (MAT_DIM * blockDim.x));
	//int flag = 0;
    for(int global_block_id = blockIdx.x; global_block_id < blocks_in_state_vector; global_block_id += gridDim.x) {
		__syncthreads();

        //inside batch0
        //int blocks_per_batch0 = (1 << qid0) / elements_per_chunk;
        smem[0] = (1 << qid0) / elements_per_chunk;
        //int batch0_stride = 2 * (1 << qid0);
        smem[1] = 2 * (1 << qid0);

        //inside batch1
        //int blocks_per_batch1 = (1 << qid1) / (2 * elements_per_chunk);
        smem[2] = (1 << qid1) / (2 * elements_per_chunk);
        //int batch1_depth = (global_block_id % smem[2]);
        smem[3] = (global_block_id % smem[2]);
        //int batch1_stride = 2 * (1 << qid1);
        smem[4] = 2 * (1 << qid1);

		//inside batch2
		//int blocks_per_batch2 = (1 << qid2) / (4 * elements_per_chunk);
		smem[5] = (1 << qid2) / (4 * elements_per_chunk);
		//int batch2_depth = (global_block_id % smem[5]);
		smem[6] = (global_block_id % smem[5]);
		//int batch2_stride = 2 * (1 << qid2);
		smem[7] = 2 * (1 << qid2);

		//inside batch3
		//int blocks_per_batch3 = (1 << qid3) / (8 * elements_per_chunk);
		smem[8] = (1 << qid3) / (8 * elements_per_chunk);
		//int batch3_depth = (global_block_id % smem[8]);
		smem[9] = (global_block_id % smem[8]);
		//int batch3_stride = 2 * (1 << qid3);
		smem[10] = 2 * (1 << qid3);

		//inside batch4
		//int blocks_per_batch4 = (1 << qid4) / (16 * elements_per_chunk);
		smem[11] = (1 << qid4) / (16 * elements_per_chunk);
		//int batch4_depth = (global_block_id % smem[11]);
		smem[12] = (global_block_id % smem[11]);
		//int batch4_stride = 2 * (1 << qid4);
		smem[13] = 2 * (1 << qid4);

        //ids
        //int chunk_id = global_block_id % smem[0];
        smem[14] = global_block_id % smem[0];
        //int batch0_id = smem[3] / smem[0];
        smem[15] = smem[3] / smem[0];
        //int batch1_id = smem[6] / smem[2];
        smem[16] = smem[6] / smem[2];
        //int batch2_id = smem[9] / smem[5];
        smem[17] = smem[9] / smem[5];
        //int batch3_id = smem[12] / smem[8];
        smem[18] = smem[12] / smem[8];
        //int batch4_id = global_block_id / smem[11];
        smem[19] = global_block_id / smem[11];

        int element_id_base = 0;
        element_id_base += threadIdx.x;
        element_id_base += smem[14] * elements_per_chunk;
        element_id_base += smem[15] * smem[1];
        element_id_base += smem[16] * smem[4];
        element_id_base += smem[17] * smem[7];
        element_id_base += smem[18] * smem[10];
        element_id_base += smem[19] * smem[13];

        //iteration dependent

#define compute(i,j,k,l,m)\
		{\
		int offset = (i * (1 << qid4)) + (j * (1 << qid3)) + (k * (1 << qid2)) + (l * (1 << qid1)) + (m * (1 << qid0));\
		int element_id = element_id_base + offset;\
		complex val = C(0.0f,0.0f);\
		if(element_id < vec_size) val = vec[element_id];\
		int column = (16*i)+(8*j)+(4*k)+(2*l)+m;\
		for(int row = 0; row < MAT_DIM; row++) result[row] = result[row] + (operator_matrix[row][column]*val);\
		}


        __shared__ complex result[MAT_DIM];
        for(int row = 0; row < MAT_DIM; row++) result[row] = C(0.0f, 0.0f);


		compute(0,0,0,0,0);
		compute(0,0,0,0,1);
		compute(0,0,0,1,0);
		compute(0,0,0,1,1);
		compute(0,0,1,0,0);
		compute(0,0,1,0,1);
		compute(0,0,1,1,0);
		compute(0,0,1,1,1);
		compute(0,1,0,0,0);
		compute(0,1,0,0,1);
		compute(0,1,0,1,0);
		compute(0,1,0,1,1);
		compute(0,1,1,0,0);
		compute(0,1,1,0,1);
		compute(0,1,1,1,0);
		compute(0,1,1,1,1);
		compute(1,0,0,0,0);
		compute(1,0,0,0,1);
		compute(1,0,0,1,0);
		compute(1,0,0,1,1);
		compute(1,0,1,0,0);
		compute(1,0,1,0,1);
		compute(1,0,1,1,0);
		compute(1,0,1,1,1);
		compute(1,1,0,0,0);
		compute(1,1,0,0,1);
		compute(1,1,0,1,0);
		compute(1,1,0,1,1);
		compute(1,1,1,0,0);
		compute(1,1,1,0,1);
		compute(1,1,1,1,0);
		compute(1,1,1,1,1);


		#define store(i,j,k,l,m) \
		{\
		int offset = (i * (1 << qid4)) + (j * (1 << qid3)) + (k * (1 << qid2)) + (l * (1 << qid1)) + (m * (1 << qid0));\
		int element_id = element_id_base + offset;\
		int row = (16*i)+(8*j)+(4*k)+(2*l)+m;\
		if(element_id < vec_size) vec[element_id] = result[row];\
		}

		store(0,0,0,0,0);
		store(0,0,0,0,1);
		store(0,0,0,1,0);
		store(0,0,0,1,1);
		store(0,0,1,0,0);
		store(0,0,1,0,1);
		store(0,0,1,1,0);
		store(0,0,1,1,1);
		store(0,1,0,0,0);
		store(0,1,0,0,1);
		store(0,1,0,1,0);
		store(0,1,0,1,1);
		store(0,1,1,0,0);
		store(0,1,1,0,1);
		store(0,1,1,1,0);
		store(0,1,1,1,1);
		store(1,0,0,0,0);
		store(1,0,0,0,1);
		store(1,0,0,1,0);
		store(1,0,0,1,1);
		store(1,0,1,0,0);
		store(1,0,1,0,1);
		store(1,0,1,1,0);
		store(1,0,1,1,1);
		store(1,1,0,0,0);
		store(1,1,0,0,1);
		store(1,1,0,1,0);
		store(1,1,0,1,1);
		store(1,1,1,0,0);
		store(1,1,1,0,1);
		store(1,1,1,1,0);
		store(1,1,1,1,1);

    }
}

//TODO: Header
template <class M>
void run_kernel(complex* vec, int vec_size, int quid0, int quid1, int quid2, int quid3, int quid4, M source_matrix) {
    cudaDeviceSynchronize();

    //Get smem size
    cudaDeviceProp deviceProp;
    int dev_id = 0;
    cudaGetDeviceProperties(&deviceProp, dev_id);
    int smem_size_in_bytes = (int) deviceProp.sharedMemPerBlock;
    int smem_size_in_elems = smem_size_in_bytes/(2*sizeof(double));

    int max_threads_per_block = (int) deviceProp.maxThreadsPerBlock;

    //batch: pairs before regions overlap
	const unsigned long batch_size = 1UL << (quid0); //in elements

	//A chunk can't be larger than shared memory because we need to hold it all at once
	//A chunk can't be larger than the threads in a block because we need one thread to handle each element
	//A chunk can't be larger than a batch by definition
    int chunk_size = std::min(std::min(smem_size_in_elems, max_threads_per_block),(int) batch_size);
	//CRK: Full blocks are allocating too many resources (not enough registers per block), so I'm nerfing it
	if (chunk_size > 512) {
		chunk_size = 512;
	}
    int chunk_size_in_bytes = chunk_size * sizeof(complex);
    dim3 blockDim(chunk_size);
	int max_grid_size = deviceProp.maxGridSize[0];
    dim3 gridDim(std::min(max_grid_size, (int) ceil(vec_size/(float)chunk_size)));

	size_t max_const_mem = deviceProp.totalConstMem;

	int max_reg_per_block = deviceProp.regsPerBlock;

    //print some stats about the GPU
    std::cout << "smem_size_in_elems: " << smem_size_in_elems << std::endl;
    std::cout << "max_threads_per_block: " << max_threads_per_block << std::endl;;
    std::cout << "batch size: " << batch_size << std::endl;;
    std::cout << "chunk size: " << chunk_size << std::endl;;
    std::cout << "max grid size: " << max_grid_size << std::endl;;
	std::cout << "const mem on device: " << max_const_mem << std::endl;
	std::cout << "max registers per block: " << max_reg_per_block << std::endl;

	std::cout << "Vec size (num vectors is log2): " << vec_size << std::endl;
	std::cout << "quid0: " << quid0 << std::endl;
	std::cout << "quid1: " << quid1 << std::endl;
	std::cout << "quid2: " << quid2 << std::endl;
	std::cout << "quid3: " << quid3 << std::endl;
	std::cout << "quid4: " << quid4 << std::endl;

    std::cout << "block dim: " << blockDim.x << std::endl;
    std::cout << "grid dim: " << gridDim.x << std::endl;

    //memcpy and run the kernel
	start = std::chrono::high_resolution_clock::now();

    complex *d_vec;
    cudaError_t malloc_error = cudaMalloc((void **) &d_vec, vec_size*sizeof(complex));
	std::cout << "Malloc error is: " << malloc_error << std::endl;
    cudaError_t cpy_error = cudaMemcpy(d_vec, vec, vec_size*sizeof(complex), cudaMemcpyHostToDevice);
	std::cout << "Copying to device error is: " << cpy_error << std::endl;
    five_qubit_kernel<<<gridDim, blockDim, chunk_size_in_bytes>>>(d_vec, vec_size, quid0, quid1, quid2, quid3, quid4, chunk_size);
    cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cout << "Kernel failed with error: " << cudaGetErrorString(err) << std::endl;
	}
    cpy_error = cudaMemcpy(vec, d_vec, vec_size*sizeof(complex), cudaMemcpyDeviceToHost);
	std::cout << "Copying to host error is: " << cpy_error << std::endl;
    cudaFree(d_vec);

	stop = std::chrono::high_resolution_clock::now();
}

int main(int argc, char **argv) {

	if (argc != 7) {
		std::cout << "Input args wrong. Needs exactly six input args" << std::endl;
		exit(1);
	}


	int quid0 = atoi(argv[2]);
	int quid1 = atoi(argv[3]);
	int quid2 = atoi(argv[4]);
	int quid3 = atoi(argv[5]);
	int quid4 = atoi(argv[6]);

    //Read state vector
    std::vector<complex> state_vec;
    std::ifstream fin;
    complex temp;
	std::complex<float> std_complex_temp;
	/*
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
	std::cout << "Vector size: " << state_vec_size << std::endl;
    fin.close();
	*/
    for (unsigned long i = 0; i < 1 << atoi(argv[1]); i++){ 
        //Note: normalization ignored for now
        float real = ((float) rand() / (float) (RAND_MAX));
        float imag = ((float) rand() / (float) (RAND_MAX));
        complex val = C(real, imag);
		state_vec.push_back(val);
    }

    unsigned long state_vec_size = state_vec.size();


    std::vector<complex> source_matrix_vec;
	std::cout << "here is the source matrix: " << std::endl;
    fin.open(MAT_FILENAME);
    while(fin >> std_complex_temp) {
		temp = C(std_complex_temp.real(), std_complex_temp.imag());
		std::cout << temp << std::endl;
		source_matrix_vec.push_back(temp);
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
    cudaError_t symbol_error = cudaMemcpyToSymbol(operator_matrix, source_matrix_vec.data(), MAT_DIM * MAT_DIM * sizeof(complex), 0, cudaMemcpyHostToDevice);
	std::cout << "Error from symbol copy is: " << symbol_error << std::endl;

    run_kernel(state_vec.data(), state_vec_size, quid0, quid1, quid2, quid3, quid4, source_matrix_vec.data());


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
