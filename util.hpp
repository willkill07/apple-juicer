#pragma once

#include <bit>
#include <chrono>

#include <sys/sysctl.h>
#include <sys/types.h>

using Clock = std::chrono::steady_clock;

constexpr double kMemoryUsageFactor{0.5};
constexpr std::size_t kGiB{1 << 30};
constexpr std::size_t kTrials{10};
constexpr std::chrono::seconds kInitialDelayTime{10};
constexpr std::chrono::seconds kPauseDelayTime{1};

template <typename T>
constexpr std::string kLabel{
    std::array{"fp8", "fp16", "fp32", "fp64"}[std::countr_zero(sizeof(T))]};

struct TimeStat {
  using ns = std::chrono::nanoseconds;

  constexpr TimeStat() noexcept = default;

  constexpr TimeStat(ns time) noexcept : total{time}, min{time}, max{time} {}

  constexpr TimeStat &operator+=(TimeStat const &other) noexcept {
    total += other.total;
    min = std::min(min, other.min);
    max = std::max(max, other.max);
    return *this;
  }

  [[nodiscard]] constexpr TimeStat
  operator+(TimeStat const &other) const noexcept {
    return TimeStat{*this} += other;
  }

  ns total{0};
  ns min{ns::max()};
  ns max{ns::min()};
};

template <typename T>
[[nodiscard]] inline std::optional<T> SysctlByName(char const *name) noexcept {
  T value;
  size_t size = sizeof(value);
  if (::sysctlbyname(name, &value, &size, NULL, 0) < 0) {
    return std::nullopt;
  } else {
    return value;
  }
}

[[nodiscard]] inline std::optional<std::int64_t>
GetTotalMemoryBytes() noexcept {
  return SysctlByName<std::int64_t>("hw.memsize");
}

[[nodiscard]] inline std::optional<std::size_t> GetTotalCPUCount() noexcept {
  return SysctlByName<std::int32_t>("hw.perflevel0.physicalcpu");
}

[[nodiscard]] inline std::optional<std::size_t> GetTotalCPUClusters() noexcept {
  auto cpu_p_max = SysctlByName<std::int32_t>("hw.perflevel0.physicalcpu_max");
  auto cpu_p_cluster = SysctlByName<std::int32_t>("hw.perflevel0.cpusperl2");
  auto cpu_e_max = SysctlByName<std::int32_t>("hw.perflevel1.physicalcpu_max");
  auto cpu_e_cluster = SysctlByName<std::int32_t>("hw.perflevel1.cpusperl2");
  if (cpu_p_max and cpu_p_cluster and cpu_e_max and cpu_e_cluster) {
    return static_cast<std::size_t>(*cpu_p_max / *cpu_p_cluster +
                                    *cpu_e_max / *cpu_e_cluster);
  } else {
    return std::nullopt;
  }
}
