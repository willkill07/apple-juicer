#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <print>
#include <ranges>
#include <thread>

#include "util.hpp"

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

template <typename Type>
void Trial(std::size_t trials, std::int64_t mem_limit,
           std::size_t total_cores) {
  std::size_t const N{mem_limit / sizeof(Type)};

  auto a{std::make_unique_for_overwrite<Type[]>(N)};
  auto b{std::make_unique_for_overwrite<Type[]>(N)};
  auto c{std::make_unique_for_overwrite<Type[]>(N)};
  std::ranges::fill_n(a.get(), N, Type(1));
  std::ranges::fill_n(b.get(), N, Type(2));
  std::ranges::fill_n(c.get(), N, Type(0));
  Type const scalar{3};

  auto const A{a.get()}, B{b.get()}, C{c.get()};

  for (std::size_t threads{1}; threads <= total_cores; ++threads) {

    std::array<TimeStat, kNumKernels> stats;
    stats.fill(TimeStat{});

    for (std::size_t trial{0}; trial < trials; ++trial) {
      std::array const times{
          Clock::now(),                                      //
          (Copy(threads, N, A, C), Clock::now()),            //
          (Scale(threads, N, scalar, B, C), Clock::now()),   //
          (Add(threads, N, A, B, C), Clock::now()),          //
          (Triad(threads, N, scalar, A, B, C), Clock::now()) //
      };
      std::array<Clock::duration, kNumKernels> d;
      std::ranges::transform(std::views::drop(times, 1), times, d.begin(),
                             std::minus{});
      std::ranges::transform(stats, d, stats.begin(), std::plus{});
      std::this_thread::sleep_for(kPauseDelayTime);
    }

    std::invoke(
        [&]<OperationType... Ops>(Values<Ops...>) {
          (..., std::invoke(
                    [&]<OperationType Op>(Value<Op>) {
                      double const bytes{Bytes<Type, Op>(N)};
                      TimeStat const &s{stats[Op]};
                      std::println(
                          "{},{},{},{:0.3f},{:0.3f},{:0.3f}", kLabel<Type>,
                          threads, Name<Op>, trials * bytes / s.total.count(),
                          bytes / s.max.count(), bytes / s.min.count());
                    },
                    Value<Ops>{}));
        },
        Values<kCopy, kScale, kAdd, kTriad>{});
  }
}

int main() {
  std::optional const memory{GetTotalMemoryBytes()};
  std::optional const cores{GetTotalCPUCount()};
  std::optional const brand{GetCPUBrandString()};
  std::optional const caps{GetArmCapabilities()};
  std::optional const model{GetModel()};
  if (not(memory and cores and brand and caps and model)) {
    return EXIT_FAILURE;
  }
  std::println("Model: {}", *model);
  std::println("Brand: {}", *brand);
  std::println("CPU cores detected: {}", *cores);
  std::println("Capabilities: {:064b}", *caps);
  std::println("Memory detected: {} GiB", *memory / kGiB);

  std::size_t const mem_limit(*memory * kMemoryUsageFactor / kNumArrays);
  std::println("");
  std::println("Allowing total memory allocations per array to be: {} GiB\n",
               mem_limit / kGiB);

  std::println("datatype,threads,function,average,minimum,maximum");
  Trial<float>(kTrials, mem_limit, *cores);
  Trial<double>(kTrials, mem_limit, *cores);
}
