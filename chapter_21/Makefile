NVCC = nvcc
NVCC_FLAGS = -O3 -rdc=true -lcudadevrt -arch=sm_86

TARGET = chapter_21
SRCS = chapter_21.cu

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
