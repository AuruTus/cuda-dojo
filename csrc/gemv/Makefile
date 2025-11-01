# Compiler and flags
NVCC = nvcc
CXX = g++
NVCC_FLAGS = -std=c++14 -O3 -Xcompiler "-Wall -Wextra"
CXX_FLAGS = -std=c++14 -O3 -Wall -Wextra

# CUDA architecture (adjust based on your GPU)
CUDA_ARCH = -arch=sm_80

# Targets
TARGET = gemv.out
SRC = gemv.cu

# Default target
all: $(TARGET)

# Main compilation target
$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) -o $(TARGET) $(SRC)