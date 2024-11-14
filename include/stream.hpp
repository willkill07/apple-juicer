#pragma once

#include <array>
#include <cstddef>
#include <string_view>

constexpr std::size_t kNumArrays{3};

enum OperationType { kCopy, kScale, kAdd, kTriad, kNumKernels };

template <typename T, OperationType Op>
  requires(Op >= 0 and Op < kNumKernels)
[[nodiscard]] constexpr double Bytes(std::size_t N) noexcept {
  return N * sizeof(T) * std::array{2.0, 2.0, 3.0, 3.0}[Op];
}

template <OperationType Op>
  requires(Op >= 0 and Op < kNumKernels)
constexpr static inline std::string_view Name =
    std::array{"copy", "scale", "add", "triad"}[Op];

template <typename T>
void Copy([[maybe_unused]] std::size_t threads, std::size_t N, T const *A,
          T *C) {
#pragma omp parallel for simd num_threads(threads)
  for (std::size_t j = 0; j < N; j++) {
    C[j] = A[j];
  }
}

template <typename T>
void Scale([[maybe_unused]] std::size_t threads, std::size_t N, T scalar, T *B,
           T const *C) {
#pragma omp parallel for simd num_threads(threads)
  for (std::size_t j = 0; j < N; j++) {
    B[j] = scalar * C[j];
  }
}

template <typename T>
void Add([[maybe_unused]] std::size_t threads, std::size_t N, T const *A,
         T const *B, T *C) {
#pragma omp parallel for simd num_threads(threads)
  for (std::size_t j = 0; j < N; j++) {
    C[j] = A[j] + B[j];
  }
}

template <typename T>
void Triad([[maybe_unused]] std::size_t threads, std::size_t N, T scalar, T *A,
           T const *B, T const *C) {
#pragma omp parallel for simd num_threads(threads)
  for (std::size_t j = 0; j < N; j++) {
    A[j] = B[j] + scalar * C[j];
  }
}