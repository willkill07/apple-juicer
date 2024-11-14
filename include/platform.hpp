#pragma once

#include <optional>
#include <string>

#include <sys/sysctl.h>
#include <sys/types.h>

namespace platform {

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

[[nodiscard]] inline std::optional<std::string> GetCPUBrandString() noexcept {
  char value[256];
  size_t size = sizeof(value);
  if (::sysctlbyname("machdep.cpu.brand_string", &value, &size, NULL, 0) < 0) {
    return std::nullopt;
  } else {
    return std::string{value, size};
  }
}

[[nodiscard]] inline std::optional<std::string> GetModel() noexcept {
  char value[256];
  size_t size = sizeof(value);
  if (::sysctlbyname("hw.model", &value, &size, NULL, 0) < 0) {
    return std::nullopt;
  } else {
    return std::string{value, size};
  }
}

[[nodiscard]] inline std::optional<std::uint64_t>
GetArmCapabilities() noexcept {
  return SysctlByName<std::uint64_t>("hw.optional.arm.caps");
}

} // namespace platform