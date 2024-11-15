#pragma once

#include "smc.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <stop_token>
#include <thread>

class PowerReader {
public:
  inline ~PowerReader() {
    stop_source.request_stop();
    runner.join();
  }

  inline void Start() {
    std::unique_lock lock{mutex};
    times.clear();
  }

  [[nodiscard]] inline float Stop() {
    static std::vector<float> copy;
    {
      std::unique_lock lock{mutex};
      copy.assign_range(times);
      times.clear();
    }
    return [this] {
      if (copy.empty()) {
        return last.load();
      } else {
        return std::ranges::fold_left(copy, 0.0f, std::plus{}) / copy.size();
      }
    }();
  }

private:
  inline void Updater() {
    std::this_thread::sleep_for(std::chrono::milliseconds{500});
    while (not stop_source.stop_requested()) {
      smc->ReadVal<float>("PSTR").transform([this](float w) {
        last.store(w);
        std::unique_lock lock{mutex};
        times.push_back(w);
        return w;
      });
      std::this_thread::sleep_for(std::chrono::seconds{1});
    }
  }

  std::optional<SMC> smc{SMC::Make()};
  std::mutex mutex;
  std::vector<float> times;
  std::atomic<float> last{0.0f};
  std::stop_source stop_source;
  std::thread runner{std::mem_fn(&PowerReader::Updater), this};
};
