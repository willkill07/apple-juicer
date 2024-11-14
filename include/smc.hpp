#pragma once

#include <optional>
#include <string_view>
#include <unordered_map>

using Key = unsigned int;
using Connection = unsigned int;

struct KeyDataVer {
  unsigned char major{0};
  unsigned char minor{0};
  unsigned char build{0};
  unsigned char reserved{0};
  unsigned short release{0};
};

struct PLimitData {
  unsigned short version{0};
  unsigned short length{0};
  unsigned int cpu_p_limit{0};
  unsigned int gpu_p_limit{0};
  unsigned int mem_p_limit{0};
};

struct KeyInfo {
  unsigned int data_size{0};
  unsigned int data_type{0};
  unsigned char data_attributes{0};
};

class SMC;

class KeyData {
  friend class SMC;

public:
  KeyData() noexcept;
  KeyData(unsigned char data8, Key key) noexcept;
  KeyData(unsigned char data8, Key key, KeyInfo key_info) noexcept;

  template <typename T> [[nodiscard]] T As() const noexcept {
    return *reinterpret_cast<T const *>(bytes);
  }

private:
  [[maybe_unused]] Key key{0};
  KeyDataVer vers{};
  PLimitData p_limit_data{};
  KeyInfo key_info{};
  unsigned char result{0};
  [[maybe_unused]] unsigned char status{0};
  [[maybe_unused]] unsigned char data8{0};
  [[maybe_unused]] unsigned int data32{0};
  char bytes[32];
};

class SMC {
  SMC(SMC const &) = delete;
  SMC &operator=(SMC const &) = delete;
  SMC(SMC &&) = delete;
  SMC &operator=(SMC &&) = delete;

  [[nodiscard]] static std::optional<Connection> GetConnection() noexcept;

  SMC(Connection conn) noexcept;

  [[nodiscard]] std::optional<KeyData>
  Read(KeyData const &input) const noexcept;

  std::optional<KeyInfo> ReadKeyInfo(Key key);

  [[nodiscard]] std::optional<KeyData> ReadValRaw(std::string_view v);

public:
  static std::optional<SMC> Make() noexcept;

  ~SMC() noexcept;

  template <typename T>
  [[nodiscard]] std::optional<T> ReadVal(std::string_view v) {
    return ReadValRaw(v).transform(
        [](KeyData &&output) { return output.As<T>(); });
  }

private:
  Connection conn;
  std::unordered_map<Key, KeyInfo> map;
};
