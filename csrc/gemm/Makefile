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

.PHONY: profile/perf-setup
profile/perf-setup:
	@echo "Setting up permissions for CPU profiling..."
	echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid
	@echo "Current perf_event_paranoid value:"
	cat /proc/sys/kernel/perf_event_paranoid

.PHONY: profile/perf-reset
profile/perf-reset:
	@echo "Resetting perf_event_paranoid to default..."
	echo 2 | sudo tee /proc/sys/kernel/perf_event_paranoid

.PHONY: profile/nsys
profile/nsys: profile/perf-setup
	nsys profile --stats=true -s cpu --trace=cuda,nvtx,osrt -o $(NSYS_REPT) ./$(TARGET)
	$(MAKE) profile/perf-reset

.PHONY: profile/ncu
profile/ncu:
	ncu --force-overwrite --set full -o $(NCU_REPT) ./$(TARGET)

