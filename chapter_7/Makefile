NVCC = nvcc
NVCC_FLAGS = -O3

TARGET = chapter_7
SOURCE = chapter_7.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean run
