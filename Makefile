CC=g++
CC_FLAGS=--std=c++11
NVCC=nvcc
KERNEL_NO_INTRIN_INC=-I./projectq/backends/_sim/_cppkernels/nointrin/
KERNEL_INTRIN_INC=-I./projectq/backends/_sim/_cppkernels/intrin/
CUDA_KERNELS=kernel*.cu

all: \
	gen_state_vec.o \
	projectq_kernel_no_intrin_runner.o \
	projectq_kernel_intrin_runner.o \
	$(CUDA_KERNELS)

gen_state_vec.o: gen_state_vec.cpp
	$(CC) $(CC_FLAGS) $^ -o $@

projectq_kernel_no_intrin_runner.o: 
	$(CC) $(CC_FLAGS) projectq_kernel_nointrin_runner.cpp -o $@ -I $(KERNEL_NO_INTRIN_INC)

projectq_kernel_intrin_runner.o: 
	$(CC) $(CC_FLAGS) -mavx projectq_kernel_intrin_runner.cpp -o $@ -I $(KERNEL_INTRIN_INC)

kernel%.o: kernel%.cu
	$(NVCC) $(CC_FLAGS) $^ -o $@
