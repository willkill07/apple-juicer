#pragma once

#include <chrono>

using Clock = std::chrono::steady_clock;
constexpr std::chrono::seconds kInitialDelayTime{10};
constexpr std::chrono::seconds kMinimumTime{4};

struct TimeStat {

  constexpr TimeStat() noexcept = default;

  template <typename Rep, typename Period>
  constexpr TimeStat(std::chrono::duration<Rep, Period> const &time) noexcept
      : count{1},
        total{std::chrono::duration_cast<std::chrono::nanoseconds>(time)},
        min{std::chrono::duration_cast<std::chrono::nanoseconds>(time)},
        max{std::chrono::duration_cast<std::chrono::nanoseconds>(time)} {}

  constexpr TimeStat &operator+=(TimeStat const &other) noexcept {
    count += other.count;
    total += other.total;
    min = std::min(min, other.min);
    max = std::max(max, other.max);
    return *this;
  }

  [[nodiscard]] constexpr TimeStat
  operator+(TimeStat const &other) const noexcept {
    return TimeStat{*this} += other;
  }

  std::size_t count{0};
  std::chrono::nanoseconds total{0};
  std::chrono::nanoseconds min{std::chrono::nanoseconds::max()};
  std::chrono::nanoseconds max{std::chrono::nanoseconds::min()};
};
