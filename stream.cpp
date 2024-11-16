#include "stream.hpp"

#include "constants.hpp"
#include "meta.hpp"
#include "platform.hpp"
#include "time.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <print>
#include <ranges>

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

    std::array stats{TimeStat{}, TimeStat{}, TimeStat{}, TimeStat{}};

    for (std::size_t trial{0}; trial < trials; ++trial) {
      std::array const times{
          Clock::now(),                                      //
          (Copy(threads, N, A, C), Clock::now()),            //
          (Scale(threads, N, scalar, B, C), Clock::now()),   //
          (Add(threads, N, A, B, C), Clock::now()),          //
          (Triad(threads, N, scalar, A, B, C), Clock::now()) //
      };
      std::ranges::transform(
          stats,
          std::views::zip(std::views::drop(times, 1), times) |
              std::views::transform([](auto&& p) { //
                return std::apply(std::minus{}, p);
              }),
          stats.begin(), std::plus{});
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
  std::optional const memory{platform::GetTotalMemoryBytes()};
  std::optional const cores{platform::GetTotalCPUCount()};
  std::optional const brand{platform::GetCPUBrandString()};
  std::optional const caps{platform::GetArmCapabilities()};
  std::optional const model{platform::GetModel()};
  if (not(memory and cores and brand and caps and model)) {
    return EXIT_FAILURE;
  }
  std::println("Model: {}", *model);
  std::println("Brand: {}", *brand);
  std::println("CPU cores detected: {}", *cores);
  std::println("Capabilities: {:064b}", *caps);
  std::println("Memory detected: {} GiB", *memory / kGiB);

  std::size_t const mem_limit(*memory * kMemoryUsageFactor / kNumArrays);
  std::println("\nAllowing total memory allocations per array to be: {} GiB\n",
               mem_limit / kGiB);

  std::println("datatype,threads,function,average,minimum,maximum");
  Trial<float>(kTrials, mem_limit, *cores);
  Trial<double>(kTrials, mem_limit, *cores);
}
