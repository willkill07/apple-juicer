CXX := clang++
LINK.o := clang++

CPPFLAGS := -I./include -I/opt/homebrew/opt/libomp/include -Xpreprocessor -fopenmp
CXXFLAGS := -O3 -march=native -std=c++2b -fexperimental-library -Wall -Wextra -Werror
LDLIBS := -L/opt/homebrew/opt/libomp/lib -lomp -framework Accelerate -framework IOKit

.PHONY: all clean

all : stream gemm

gemm : gemm.o smc.o

clean :
	-rm -vf stream gemm stream.o gemm.o smc.o
