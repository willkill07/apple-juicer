CXX := clang++

CPPFLAGS := -I/opt/homebrew/opt/libomp/include -Xpreprocessor -fopenmp
CXXFLAGS := -O3 -march=native -std=c++2b -Wall -Wextra -Werror
LDLIBS := -L/opt/homebrew/opt/libomp/lib -lomp -framework Accelerate

.PHONY: all clean

all : stream gemm

clean :
	-rm -vf stream gemm stream.o gemm.o
