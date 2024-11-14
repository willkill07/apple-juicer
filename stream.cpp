#include <algorithm>
#include <chrono>
#include <memory>
#include <print>
#include <thread>

#include <omp.h>

#include "util.hpp"

template <typename T>
void copy([[maybe_unused]] std::size_t threads, std::size_t N, T const *a,
          T *c) {
#pragma omp parallel for simd num_threads(threads)
  for (std::size_t j = 0; j < N; j++) {
    c[j] = a[j];
  }
}

template <typename T>
void scale([[maybe_unused]] std::size_t threads, std::size_t N, T scalar, T *b,
           T const *c) {
#pragma omp parallel for simd num_threads(threads)
  for (std::size_t j = 0; j < N; j++) {
    b[j] = scalar * c[j];
  }
}

template <typename T>
void add([[maybe_unused]] std::size_t threads, std::size_t N, T const *a,
         T const *b, T *c) {
#pragma omp parallel for simd num_threads(threads)
  for (std::size_t j = 0; j < N; j++) {
    c[j] = a[j] + b[j];
  }
}

template <typename T>
void triad([[maybe_unused]] std::size_t threads, std::size_t N, T scalar, T *a,
           T const *b, T const *c) {
#pragma omp parallel for simd num_threads(threads)
  for (std::size_t j = 0; j < N; j++) {
    a[j] = b[j] + scalar * c[j];
  }
}

template <typename Type>
void RunTrials(std::size_t trials, std::int64_t total_memory,
               std::size_t total_cores) {
  constexpr std::size_t kNumArrays{3};
  auto const N = static_cast<std::size_t>(total_memory * kMemoryUsageFactor) /
                 (sizeof(Type) * kNumArrays);

  auto a = std::make_unique_for_overwrite<Type[]>(N);
  auto b = std::make_unique_for_overwrite<Type[]>(N);
  auto c = std::make_unique_for_overwrite<Type[]>(N);

  std::ranges::fill_n(a.get(), N, Type(1));
  std::ranges::fill_n(b.get(), N, Type(2));
  std::ranges::fill_n(c.get(), N, Type(0));
  Type const scalar{3};

  for (std::size_t threads = 1zu; threads <= total_cores; ++threads) {

    TimeStat copy_stat{};
    TimeStat scale_stat{};
    TimeStat add_stat{};
    TimeStat triad_stat{};

    for (std::size_t trial = 0; trial < trials; ++trial) {
      std::array<typename Clock::time_point, 5> t;
      t[0] = Clock::now();
      copy(threads, N, a.get(), c.get());
      t[1] = Clock::now();
      scale(threads, N, scalar, b.get(), c.get());
      t[2] = Clock::now();
      add(threads, N, a.get(), b.get(), c.get());
      t[3] = Clock::now();
      triad(threads, N, scalar, a.get(), b.get(), c.get());
      t[4] = Clock::now();
      copy_stat +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(t[1] - t[0]);
      scale_stat +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(t[2] - t[1]);
      add_stat +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(t[3] - t[2]);
      triad_stat +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(t[4] - t[3]);
      std::this_thread::sleep_for(kPauseDelayTime);
    }

    double const copy_bytes{N * sizeof(Type) * 2.0};
    double const scale_bytes{N * sizeof(Type) * 2.0};
    double const add_bytes{N * sizeof(Type) * 3.0};
    double const triad_bytes{N * sizeof(Type) * 3.0};

    std::println("{},{},{},{:0.3f},{:0.3f},{:0.3f}", kLabel<Type>, threads,
                 "copy", trials * copy_bytes / copy_stat.total.count(),
                 copy_bytes / copy_stat.max.count(),
                 copy_bytes / copy_stat.min.count());
    std::println("{},{},{},{:0.3f},{:0.3f},{:0.3f}", kLabel<Type>, threads,
                 "scale", trials * scale_bytes / scale_stat.total.count(),
                 scale_bytes / scale_stat.max.count(),
                 scale_bytes / scale_stat.min.count());
    std::println("{},{},{},{:0.3f},{:0.3f},{:0.3f}", kLabel<Type>, threads,
                 "add", trials * add_bytes / add_stat.total.count(),
                 add_bytes / add_stat.max.count(),
                 add_bytes / add_stat.min.count());
    std::println("{},{},{},{:0.3f},{:0.3f},{:0.3f}", kLabel<Type>, threads,
                 "triad", trials * triad_bytes / triad_stat.total.count(),
                 triad_bytes / triad_stat.max.count(),
                 triad_bytes / triad_stat.min.count());
  }
}

int main() {

  std::optional total_memory = GetTotalMemoryBytes();
  std::optional total_cores = GetTotalCPUCount();

  std::println("Detected {} CPU Cores\n", *total_cores);

  std::println("datatype,threads,function,average,minimum,maximum");
  RunTrials<float>(kTrials, *total_memory, *total_cores);
  RunTrials<double>(kTrials, *total_memory, *total_cores);
}
