# Compile for your GPU architecture (check with nvidia-smi)
all:
	nvcc template.cu -o my_program.out -arch=sm_89 -O3

# Run the program
# ===================
# ./my_program
# ===================

# Check GPU architecture
# ===================
# nvidia-smi --query-gpu=compute_cap --format=csv
# ===================