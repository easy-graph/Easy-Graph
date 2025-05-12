# Compiler
NVCC = nvcc

# Compiler Flags
CFLAGS = -O3 -std=c++14 -arch=sm_70

# Target executable
TARGET = ecl_mst

# Source file
SRC = ECL-MST_10.cu

# Build rule
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) $(SRC) -o $(TARGET)

# Clean rule
clean:
	rm -f $(TARGET)
