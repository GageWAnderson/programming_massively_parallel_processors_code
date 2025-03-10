NVCC = nvcc
NVCC_FLAGS = -O3

TARGET = chapter_12
SRCS = chapter_12.cu

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
