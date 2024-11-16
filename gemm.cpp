#include "gemm.hpp"

#include "constants.hpp"
#include "meta.hpp"
#include "platform.hpp"
#include "power.hpp"
#include "time.hpp"

#include <optional>
#include <print>
#include <thread>

#include <cstdlib>

template <typename T>
void Trial(PowerReader &power_reader, std::size_t mem_limit) {

  std::vector const sizes{GenerateSizes<T>(mem_limit)};
  GemmData<T> data{sizes.back()};

  std::this_thread::sleep_for(kInitialDelayTime);
  for (std::size_t const n : sizes) {
    TimeStat stat;
    auto const start_time{Clock::now()};
    power_reader.Start();
    while (true) {
      auto const start{Clock::now()};
      Gemm(data, n);
      auto const stop{Clock::now()};
      stat += TimeStat{stop - start};
      if ((stop - start_time) >= kMinimumTime) {
        break;
      }
    }
    float const power{power_reader.Stop()};
    double const ops{n * n * n * 2.0};
    std::println("{},{},{:0.3f},{:0.3f},{:0.3f},{:0.3f},{:0.3f}", //
                 kLabel<T>, n,                                    //
                 ops * stat.count / stat.total.count(),           //
                 ops / stat.max.count(),                          //
                 ops / stat.min.count(),                          //
                 power,                                           //
                 ops * stat.count / stat.total.count() / power);
  }
}

int main() {

  std::optional const memory{platform::GetTotalMemoryBytes()};
  std::optional const cores{platform::GetTotalCPUCount()};
  std::optional const brand{platform::GetCPUBrandString()};
  std::optional const caps{platform::GetArmCapabilities()};
  std::optional const model{platform::GetModel()};
  std::optional const clusters{platform::GetTotalCPUClusters()};
  if (not(memory and cores and brand and caps and model and clusters)) {
    return EXIT_FAILURE;
  }
  std::println("Model: {}", *model);
  std::println("Brand: {}", *brand);
  std::println("CPU cores detected: {}", *cores);
  std::println("Capabilities: {:064b}", *caps);
  std::println("Memory detected: {} GiB", *memory / kGiB);
  std::println("CPU clusters detected: {}", *clusters);

  std::size_t const mem_limit(kMemoryUsageFactor * *memory);
  std::println("\nAllowing total memory allocations per matrix to be: {} GiB\n",
               mem_limit / (kNumMatrices * kGiB));

  std::println("datatype,size,average,minimum,maximum,power,efficiency");

  PowerReader power_reader;

  Trial<float>(power_reader, mem_limit);
  Trial<double>(power_reader, mem_limit);
}
