NVCC = nvcc
NVCC_FLAGS = -O3

TARGET = chapter_9
SRCS = chapter_9.cu

$(TARGET): $(SRCS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

.PHONY: run clean

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
