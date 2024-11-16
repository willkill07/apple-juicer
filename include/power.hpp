#pragma once

#include "smc.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <stop_token>
#include <thread>

class PowerReader {
public:
  inline PowerReader() : runner{std::bind_front(&PowerReader::Update, this)} {}

  inline ~PowerReader() { runner.request_stop(); }

  inline void Start() {
    std::unique_lock lock{mutex};
    times.clear();
  }

  [[nodiscard]] inline float Stop() {
    std::unique_lock lock{mutex};
    if (times.empty()) {
      return last;
    } else {
      float const time{std::ranges::fold_left(times, 0.0f, std::plus{}) / times.size()};
      times.clear();
      return time;
    }
  }

private:
  void Update(std::stop_token token) {
    std::this_thread::sleep_for(std::chrono::milliseconds{500});
    while (not token.stop_requested()) {
      if (auto watts = smc->ReadVal<float>("PSTR"); watts) {
        std::unique_lock lock{mutex};
        last = *watts;
        times.push_back(*watts);
      }
      std::this_thread::sleep_for(std::chrono::seconds{1});
    }
  }

  std::optional<SMC> smc{SMC::Make()};

  std::mutex mutex;
  std::vector<float> times;
  float last{0.0f};

  std::jthread runner;
};
