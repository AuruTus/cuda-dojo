NVCC = nvcc
CFLAGS = -arch=sm_89 -O3

SOURCE = mat_mul.cu
TARGET = mat_mul.out

NSYS_REPT = nsys_mat_mul_report
NCU_REPT = ncu_mat_mul_report

NSYS_EXPORT_FILE = nsys_mat_mul_report.json
NCU_EXPORT_FILE = ncu_mat_mul_report.csv


$(TARGET): $(SOURCE)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SOURCE)

.PHONY: clean
clean:
	rm -f $(TARGET)
	rm -f $(NSYS_REPT).sqlite $(NSYS_REPT).nsys-rep
	rm -f $(NCU_REPT).ncu-rep
	rm -f $(NSYS_EXPORT_FILE) $(NCU_EXPORT_FILE)

.PHONY: profile/nsys
profile/nsys:
	nsys profile --stats=true -s cpu --trace=cuda,nvtx,osrt -o $(NSYS_REPT) ./$(TARGET)

.PHONY: profile/ncu
profile/ncu:
	ncu -o $(NCU_REPT) ./$(TARGET)
