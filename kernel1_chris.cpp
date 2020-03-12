//TODO: I got this off of stackoverflow. I don't know if we actually have thrust
#include <thrust/complex.h>

#define MAT_DIM = 2;
typedef thrust::complex<float> complex;

__constant__ complex operator_matrix[MAT_DIM][MAT_DIM];

__global__ void one_qubit_kernel(complex* vec, int vec_size, int qubit_id, int elements_per_chunk) {

    //batch: pairs before regions overlap
    const int batch_size = pow(2,quibit_id+1); //in elements
    
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
            int thread_in_first_half = thread_id < threads_per_block/2;
            //The first half of the chunk contains elements from one place in memory. The second half from another place in memory.
            int offset = (thread_in_first_half) * (batch_size/2);
        
            //Load in your element (state in the state_vec)
            //First index to a batch
            //Then, index into a chunk in that batch. The chunks read in data separate by chunk_size/2
            //Use offset to account for the fact that half of the threads read from the first half of the batch, and the other threads read from the second half of the batch
            //Finally, use the thread_id to figure out which element in that half-working-set to read
            int element_id = (batch_id * batch_size) + (chunk_id * (elements_per_chunk/2)) + offset + thread_id;

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
            complex result = 0;
            //Do the matrix multiplication. If you're dealing with the first element in the pair, you deal with the top row of the matrix. Similar for second element in pair.
            for (int i = 0; i < MAT_DIM; i++) {
                result += operator_matrix[thread_in_first_half][i] * smem[pair_indices[i]];
            }

            //Every thread stores their result back into the vector
            vec[element_id] = result;
        }
    }
}

//TODO: Header
void run_kernel(complex* vec, int vec_size, int qubit_id, complex* source_matrix) {
    cudaDeviceSynchronize());

    //Fill operator matrix in const mem
    cudaMemcpyToSymbol(operator_matrix, source_matrix, MAT_DIM * MAT_DIM * sizeof(complex), 0, cudaMemcpyDeviceToDevice);

    //Get smem size
    cudaDeviceProp deviceProp;
    int dev_id = 0;
    cudaGetDeviceProperties(&deviceProp, dev_id);
    int smem_size_in_bytes = (int) deviceProp.sharedMemPerBlock;
    int smem_size_in_elems = smem_size_in_bytes/(2*sizeof(float));

    int max_threads_per_block = (int) deviceProp.maxThreadsPerBlock;

    int chunk_size = std::min(smem_size_in_elems, max_threads_per_block);
    int chunk_size_in_bytes = chunk_size * sizeof(complex);
    dim3 blockDim(chunk_size);
    dim3 gridDim(deviceProp.maxGridSize/blockDim.x);

    one_qubit_kernel<<<gridDim, blockDim, chunk_size_in_bytes>>>(vec, vec_size, qubit_id, chunk_size);
    cudaDeviceSynchronize());
}


int main() {

    int num_qubits = 14;
    int kth_qubit = 11;

    //Generate state vector
    unsigned long state_vec_size = 1UL << num_qubits;
    std::vector<complex> state_vec(state_vec_size, 0);
    for (unsigned long i = 0; i < state_vec_size; i++) {
        float bit = i%2;
        complex real_part(bit, 0.0f);
        complex imag_part (0.0f, 1.0f-bit);
        complex val = real_part + imag_part;
        (0 + i1) or (1 + i0)
        state_vec[i] = val; 
    }

    //Generate source matrix
    complex source_matrix[4];
    source_matrix[0] = 0;
    source_matrix[1] = 1;
    source_matrix[2] = 1;
    source_matrix[3] = 0;

    //Apply NOT gate
    run_kernel(state_vec.begin(), state_vec_size, kth_qubit, source_matrix) {

    //Check state vector
    for (unsigned long i = 0; i < state_vec_size; i++) {
        float bit = i%2;
        complex correct_real(0.0f, 1.0f-bit);
        complex correct_imag(1.0f-bit, 0.0f);
        complex correct_val = correct_real + correct_image;
        assert(state_vec[i] == correct_val);
        if (state_vec[i] != correct_val) {
            cout << "state_vec[" << i << "]: " << state_vec[i] << "\t correct_val: " << correct_val << endl;
            throw std::runtime_error("Bad final val in state vec");
        }
    }

    return 0;
}