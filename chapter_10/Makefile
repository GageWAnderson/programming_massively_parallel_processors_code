NVCC = nvcc
NVCC_FLAGS = -O3

TARGET = chapter_10
SRCS = chapter_10.cu

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
