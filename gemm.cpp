#include <algorithm>
#include <barrier>
#include <memory>
#include <optional>
#include <print>
#include <random>
#include <thread>

#include <cstdlib>

#include "util.hpp"

#define ACCELERATE_NEW_LAPACK 1
#define ACCELERATE_LAPACK_ILP64
#include <Accelerate/Accelerate.h>
#include <vecLib/vecLib.h>

constexpr std::array kBases{2, 3, 5, 6, 7, 10};

consteval bool
ValidateBases(std::ranges::random_access_range auto rng) noexcept {
  return std::ranges::none_of(
      rng, [&](auto v) { return std::ranges::find(rng, v * v) != rng.end(); });
}

static_assert(ValidateBases(kBases));

template <typename Type> struct GemmData {
  std::size_t elems;
  std::unique_ptr<Type[]> a;
  std::unique_ptr<Type[]> b;
  std::unique_ptr<Type[]> c;
  Type alpha;
  Type beta;

  GemmData(std::size_t max_n)
      : elems{max_n * max_n}, a{std::make_unique_for_overwrite<Type[]>(elems)},
        b{std::make_unique_for_overwrite<Type[]>(elems)},
        c{std::make_unique_for_overwrite<Type[]>(elems)} {
    auto gen{[] {
      static std::minstd_rand rng{std::random_device{}()};
      static std::uniform_real_distribution<Type> dist(-1, 1);
      return dist(rng);
    }};
    alpha = gen();
    beta = gen();
    std::ranges::generate_n(a.get(), elems, gen);
    std::ranges::generate_n(b.get(), elems, gen);
    std::ranges::generate_n(c.get(), elems, gen);
  }

  GemmData(GemmData const &other) : GemmData(other.elems) {}
  GemmData &operator=(GemmData const &) = delete;
  GemmData(GemmData &&) noexcept = default;
  GemmData &operator=(GemmData &&) noexcept = default;
};

template <typename Type, typename Idx>
void Gemm(GemmData<Type> const &data, Idx n) {
  if constexpr (std::same_as<Type, double>) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, data.alpha,
                data.a.get(), n, data.b.get(), n, data.beta, data.c.get(), n);
  } else if constexpr (std::same_as<Type, float>) {
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, data.alpha,
                data.a.get(), n, data.b.get(), n, data.beta, data.c.get(), n);
  }
}

template <typename T>
[[nodiscard]] std::vector<std::size_t> GenerateSizes(std::size_t maximum) {
  std::size_t const max_elems{maximum / (3 * sizeof(T))};
  std::vector<std::size_t> res;
  for (std::size_t base : kBases) {
    for (std::size_t dim{base}; dim * dim <= max_elems; dim *= base) {
      res.push_back(dim);
    }
  }
  std::ranges::sort(res);
  return res;
}

template <typename T>
void Trial(std::size_t total_clusters, std::size_t mem_limit) {

  std::vector const sizes{GenerateSizes<T>(mem_limit)};

  std::vector<GemmData<T>> datas;
  for (std::size_t i{0}; i < total_clusters; ++i) {
    datas.emplace_back(sizes.back());
  }

  std::println("datatype,clusters,size,average,minimum,maximum");

  for (std::size_t clusters{1}; clusters <= total_clusters; ++clusters) {

    std::barrier<> sync(clusters);
    std::vector<std::thread> threads;
    threads.reserve(clusters);

    std::atomic_size_t index{0};
    std::atomic<TimeStat> stat{};

    for (std::size_t tid{0}; tid < clusters; ++tid) {
      threads.emplace_back([&, tid] {
        std::this_thread::sleep_for(kInitialDelayTime);

        for (std::size_t const n : sizes) {

          TimeStat local_stat;

          // we need all worker threads to start at the same time
          sync.arrive_and_wait();

          while (true) {
            if (std::size_t i{index.fetch_add(1)}; i < kTrials * clusters) {
              i %= clusters;
              auto const start{Clock::now()};
              Gemm(datas[i], n);
              auto const stop{Clock::now()};
              local_stat += TimeStat{stop - start};
            } else {
              break;
            }
          }

          // atomic update
          {
            TimeStat curr{stat.load(std::memory_order_relaxed)};
            while (!stat.compare_exchange_weak(curr, curr + local_stat,
                                               std::memory_order_release,
                                               std::memory_order_relaxed))
              ;
          }

          // must sync for atomic instructions being entirely completed
          sync.arrive_and_wait();

          if (tid == 0) {
            double const ops{n * n * n * 2.0};
            double const total_ops{ops * clusters * kTrials};
            std::println("{},{},{},{:0.3f},{:0.3f},{:0.3f}",    //
                         kLabel<T>, clusters, n,                //
                         total_ops / stat.load().total.count(), //
                         ops / stat.load().max.count(),         //
                         ops / stat.load().min.count());

            // reset all atomics to their initial values
            index.store(0);
            stat.store(TimeStat{});
          }
          std::this_thread::sleep_for(kPauseDelayTime);
        }
      });
    }

    std::ranges::for_each(threads, &std::thread::join);
    threads.clear();
  }
}

int main() {

  std::optional const memory{GetTotalMemoryBytes()};
  std::optional const cores{GetTotalCPUCount()};
  std::optional const brand{GetCPUBrandString()};
  std::optional const caps{GetArmCapabilities()};
  std::optional const model{GetModel()};
  std::optional const clusters{GetTotalCPUClusters()};
  if (not(memory and cores and brand and caps and model and clusters)) {
    return EXIT_FAILURE;
  }
  std::println("Model: {}", *model);
  std::println("Brand: {}", *brand);
  std::println("CPU cores detected: {}", *cores);
  std::println("Capabilities: {:064b}", *caps);
  std::println("Memory detected: {} GiB", *memory / kGiB);
  std::println("CPU clusters detected: {}", *clusters);

  std::size_t const mem_limit(kMemoryUsageFactor * *memory / *clusters);
  std::println("");
  std::println(
      "Allowing total memory allocations per cluster to add to: {} GiB\n",
      mem_limit / kGiB);

  Trial<float>(*clusters, mem_limit);
  Trial<double>(*clusters, mem_limit);
}
