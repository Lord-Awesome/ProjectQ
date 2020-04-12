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

    //batch: pairs before regions overlap
	const unsigned long batch_size = 1UL << (qubit_id + 1); //in elements
    
    //chunk: a portion of the batch that fits in one working set (batch size / working memory)
    const int chunks_per_batch = ceil(batch_size / (float)elements_per_chunk);

    //One chunk per block
    int total_chunks = ceil(vec_size / (float)elements_per_chunk);

    //Initialize shared memory
    extern __shared__ complex smem[];

    int blocks_per_grid = gridDim.x;
    int total_grid_sections = ceil(total_chunks / (float) blocks_per_grid);

    //In case there are more chunks than blocks in the grid
    for (int grid_section = 0; grid_section < total_grid_sections; grid_section++) {
        //Which batch within the state vector you are in
        int batch_id = ((grid_section * blocks_per_grid) + blockIdx.x) / chunks_per_batch;
        //Which chunk within the batch you are in
        int chunk_id = blockIdx.x % chunks_per_batch;
        //Which thread within the block you are in
        int threads_per_block = blockDim.x; //also equal to chunk_size

        for (int thread_id = threadIdx.x; thread_id < elements_per_chunk; thread_id += threads_per_block) {

            //Assuming high qubits, the block has data from two sections of the vector
            bool thread_in_first_half = thread_id < threads_per_block/2;
            //The first half of the chunk contains elements from one place in memory. The second half from another place in memory.
            int offset = (!thread_in_first_half) * (batch_size/2);
        
            //Load in your element (state in the state_vec)
            //First index to a batch
            //Then, index into a chunk in that batch. The chunks read in data separate by chunk_size/2
            //Use offset to account for the fact that half of the threads read from the first half of the batch, and the other threads read from the second half of the batch
            //Finally, use the thread_id to figure out which element in that half-working-set to read
            int element_id = (batch_id * batch_size) + (chunk_id * (elements_per_chunk/2)) + offset + (thread_id % (threads_per_block/2));

            __syncthreads();

            //Do the read. With all threads in the block participating, this fills smem for that block.
            if (element_id < vec_size) {
                smem[thread_id] = vec[element_id];
            }

            __syncthreads();


            //Matrix multiplication
            //Every thread is responsible for one of the output elements.
            //This is a bit messy. I was trying to make it scalable to tuples larger than pairs
            //The two elements of the pair are disjoint in smem. Figure out where they are.
            //Two threads are searching for the same pair, since the multiplication produces a 2-element column vector
            //Therefore, need to do this modulo stuff
            int pair_indices[MAT_DIM];
            pair_indices[0] = thread_id % (elements_per_chunk/2);
            pair_indices[1] = pair_indices[0] + (elements_per_chunk/2);
            complex result = C(0.0f,0.0f);
            //Do the matrix multiplication. If you're dealing with the first element in the pair, you deal with the top row of the matrix. Similar for second element in pair.
            for (int i = 0; i < MAT_DIM; i++) {
                result = result + operator_matrix[!thread_in_first_half][i] * smem[pair_indices[i]];
                //result = result + operator_matrix[thread_in_first_half][i] * C(1.0f,0.0f);
            }

            //Every thread stores their result back into the vector
			if (element_id < vec_size) {
				vec[element_id] = result;
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
V &psi, unsigned id0, M const& m, std::size_t ctrlmask)
template <class V, class M>
void run_kernel(complex* vec, int qubit_id, int vec_size, M source_matrix) {
    cudaDeviceSynchronize();

    //Get smem size
    cudaDeviceProp deviceProp;
    int dev_id = 0;
    cudaGetDeviceProperties(&deviceProp, dev_id);
    int smem_size_in_bytes = (int) deviceProp.sharedMemPerBlock;
    int smem_size_in_elems = smem_size_in_bytes/(2*sizeof(double));

    int max_threads_per_block = (int) deviceProp.maxThreadsPerBlock;

    //batch: pairs before regions overlap
	const unsigned long batch_size = 1UL << (qubit_id + 1); //in elements

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

	if (argc != 2) {
		std::cout << "Input args wrong. Needs exactly one input arg which is the kth qubit" << std::endl;
		exit(1);
	}


	int kth_qubit = atoi(argv[1]);
    //int kth_qubit = 11;

    //Read state vector
    std::vector<complex> state_vec;
    std::ifstream fin;
    fin.open(FILENAME);
    complex temp;
	std::complex<float> std_complex_temp;
    while(fin >> std_complex_temp) {
		temp = C(std_complex_temp.real(), std_complex_temp.imag());
        state_vec.push_back(temp);
    }
    state_vec.push_back(temp);
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
    source_matrix_vec.push_back(temp);
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
