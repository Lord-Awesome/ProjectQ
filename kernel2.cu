//todo: i got this off of stackoverflow. i don't know if we actually have thrust
#include <cuComplex.h>
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
#define MAT_DIM 4
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

__global__ void two_qubit_kernel(complex* vec, int vec_size, int qid0, int qid1, int elements_per_chunk) {
    //qid0 is smaller than qid1

    //batch: pairs before regions overlap
    const unsigned long batch0_size = 1UL << (qid0 + 1);
    const unsigned long batch1_size = 1UL << (qid1 + 1);
	const unsigned long batch0s_per_batch1 = 1UL << (qid1-qid0);
    
    //chunk: a portion of the batch that fits in one working set (batch size / working memory)
    const int chunks_per_batch0 = ceil(batch0_size / (float)elements_per_chunk);
    const int chunks_per_batch1 = ceil(batch1_size / (float)elements_per_chunk);

    //One chunk per block
    int total_chunks = ceil(vec_size / (float)elements_per_chunk);

    //Initialize shared memory
    extern __shared__ complex smem[];

    int blocks_per_grid = gridDim.x;
    int total_grid_sections = ceil(total_chunks / (float) 2*blocks_per_grid);

    //In case there are more chunks than blocks in the grid
    for (int grid_section = 0; grid_section < total_grid_sections; grid_section++) {
        //Which batch within the state vector you are in
		/*
        int global_batch_id = ((grid_section * blocks_per_grid) + blockIdx.x) / chunks_per_batch;
        int batch_id0 = global_batch_id % (1UL << (qid1-qid0));
        int batch_id1 = global_batch_id / (1UL << (qid1-qid0));
		*/

        int global_chunk_id = ((grid_section * blocks_per_grid) + blockIdx.x);
		int batch_id1 = global_chunk_id / (chunks_per_batch1/2);
		int batch_id0 = (global_chunk_id % (chunks_per_batch1/2)) / (chunks_per_batch0);
        int chunk_id = global_chunk_id % chunks_per_batch0;

        //Which thread within the block you are in
        int threads_per_block = blockDim.x; //also equal to chunk_size

        complex result[2]; 
        int element_id[2];
		volatile int debug_counter = 0;
        for (int thread_id = threadIdx.x; thread_id < elements_per_chunk; thread_id += threads_per_block) {
            result[0] = C(0.0f,0.0f);
            result[1] = C(0.0f,0.0f);
            for (int pair_id = 0; pair_id < 2; pair_id++) {

                bool thread_in_first_half = thread_id < threads_per_block/2;
                int offset = ((!thread_in_first_half) * (batch0_size/2)) + (pair_id*(batch1_size/2));
 
                element_id[pair_id] = (batch_id1 * batch1_size) + (batch_id0 * batch0_size);
                element_id[pair_id] += (chunk_id * (elements_per_chunk/2)) + offset + (thread_id % (threads_per_block/2));
				//element_id[pair_id] = 0;

                __syncthreads();

                //Do the read. With all threads in the block participating, this fills smem for that block.
                if (element_id[pair_id] < vec_size) {
                    smem[thread_id] = vec[element_id[pair_id]];
                }

                __syncthreads();


                //Matrix multiplication
                //Every thread is responsible for one of the output elements.
                //This is a bit messy. I was trying to make it scalable to tuples larger than pairs
                //The two elements of the pair are disjoint in smem. Figure out where they are.
                //Two threads are searching for the same pair, since the multiplication produces a 2-element column vector
                //Therefore, need to do this modulo stuff
                int pair_indices[2];
                pair_indices[0] = thread_id % (elements_per_chunk/2);
                pair_indices[1] = pair_indices[0] + (elements_per_chunk/2);
                //Do the matrix multiplication. If you're dealing with the first element in the pair, you deal with the top row of the matrix. Similar for second element in pair.
                for (int i = 0; i < 2; i++) {
                    result[0] = result[0] + operator_matrix[!thread_in_first_half][i + (pair_id*2)] * smem[pair_indices[i]];
                    result[1] = result[1] + operator_matrix[!thread_in_first_half + 2][i + (pair_id*2)] * smem[pair_indices[i]];
                }
            }

            //Every thread stores their result back into the vector
            for(int i = 0; i < 2; i++) {
                if (element_id[i] < vec_size) {
                    //vec[element_id[i]] = result[i];
                    //vec[element_id[i]] = C((float)pair_indices[0], (float)pair_indices[1]);
					if (i == 0) {
						vec[element_id[0]] = C((float)global_chunk_id, (float)batch_id0);
					}
                }
            }
/*
			//DEBUG
			if (batch_id == 0 && grid_section == 0) {
				//vec[thread_id] = smem[thread_id];
				vec[thread_id] = C((float)offset, (float) element_id);
			}
			else if (element_id < vec_size) {
				vec[element_id] = C(0.0f, 0.0f);
			}
*/
        }
    }

    //assert(vec[vec_size-1] == C(0.0f,0.0f));
}

//TODO: Header
template <class M>
void run_kernel(complex* vec, int vec_size, int quid0, int quid1, M source_matrix) {
    cudaDeviceSynchronize();

    //Fill operator matrix in const mem
    cudaMemcpyToSymbol(operator_matrix, source_matrix, MAT_DIM * MAT_DIM * sizeof(complex), 0, cudaMemcpyHostToDevice);

    //Get smem size
    cudaDeviceProp deviceProp;
    int dev_id = 0;
    cudaGetDeviceProperties(&deviceProp, dev_id);
    int smem_size_in_bytes = (int) deviceProp.sharedMemPerBlock;
    int smem_size_in_elems = smem_size_in_bytes/(2*sizeof(double));

    int max_threads_per_block = (int) deviceProp.maxThreadsPerBlock;

    //batch: pairs before regions overlap
	const unsigned long batch_size = 1UL << (quid0 + 1); //in elements

	//A chunk can't be larger than shared memory because we need to hold it all at once
	//A chunk can't be larger than the threads in a block because we need one thread to handle each element
	//A chunk can't be larger than a batch by definition
    int chunk_size = std::min(std::min(smem_size_in_elems, max_threads_per_block),(int) batch_size);
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

    std::cout << "block dim: " << blockDim.x << std::endl;
    std::cout << "grid dim: " << gridDim.x << std::endl;

    //memcpy and run the kernel
	start = std::chrono::high_resolution_clock::now();

    complex *d_vec;
    cudaMalloc((void **) &d_vec, vec_size*sizeof(complex));
    cudaError_t cpy_error = cudaMemcpy(d_vec, vec, vec_size*sizeof(complex), cudaMemcpyHostToDevice);
	std::cout << "Copying to device error is: " << cpy_error << std::endl;
    two_qubit_kernel<<<gridDim, blockDim, chunk_size_in_bytes>>>(d_vec, vec_size, quid0, quid1, chunk_size);
    cudaDeviceSynchronize();
    cpy_error = cudaMemcpy(vec, d_vec, vec_size*sizeof(complex), cudaMemcpyDeviceToHost);
	std::cout << "Copying to host error is: " << cpy_error << std::endl;
    cudaFree(d_vec);

	stop = std::chrono::high_resolution_clock::now();
}

int main(int argc, char **argv) {

	if (argc != 3) {
		std::cout << "Input args wrong. Needs exactly two input arg which is the jth and kth qubit" << std::endl;
		exit(1);
	}


	int quid0 = atoi(argv[1]);
	int quid1 = atoi(argv[2]);

    //Read state vector
    std::vector<complex> state_vec;
    std::ifstream fin;
    fin.open(FILENAME);
    complex temp;
    while(fin >> temp) {
        state_vec.push_back(temp);
    }
    state_vec.push_back(temp);
    fin.close();

    unsigned long state_vec_size = state_vec.size();


	//Read in source matrix
	complex source_matrix[4][4];
	fin.open(MAT_FILENAME);
	if (!fin.is_open()) {
		std::cout << "Source matrix does not exist!" << std::endl;
		exit(1);
	}
	std::cout << "here is the source matrix: " << std::endl;
	for (int i = 0; i < MAT_DIM; i++) {
		for (int j = 0; j < MAT_DIM; j++) {
			fin >> temp;
			source_matrix[i][j] = temp;
			std::cout << temp << std::endl;
		}
	}
	fin.close();

    //Apply gate
    run_kernel(state_vec.data(), state_vec_size, quid0, quid1, source_matrix);


	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
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
