g++ gen_state_vec.cpp -o gen_state_vec.o || exit 1
g++ projectq_kernel.cpp kernels.hpp -o projectq_kernel.o || exit 1
nvcc kernel1.cu -o kernel1.o || exit 1

./gen_state_vec.o || exit 1
./projectq_kernel.o || exit 1
./kernel1.o
