#include "smc.hpp"

#include <bit>

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/IOTypes.h>

SMC::SMC(Connection conn) noexcept : conn{conn} {}

std::optional<SMC> SMC::Make() noexcept {
  return GetConnection().transform(
      [](Connection c) { return SMC{std::move(c)}; });
}

SMC::~SMC() noexcept { IOServiceClose(conn); }

std::optional<Connection> SMC::GetConnection() noexcept {
  CFMutableDictionaryRef service = IOServiceMatching("AppleSMC");
  io_iterator_t iter;
  IOServiceGetMatchingServices(0, service, &iter);
  while (true) {
    if (io_object_t next = IOIteratorNext(iter); next == 0) {
      break;
    } else if (char n[128]; IORegistryEntryGetName(next, n) != 0) {
      break;
    } else if (std::string_view name(n); name != "AppleSMCKeysEndpoint") {
      continue;
    } else if (unsigned int c;
               IOServiceOpen(next, mach_task_self(), 0, &c) != 0) {
      break;
    } else {
      return c;
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<KeyData> SMC::ReadValRaw(std::string_view v) {
  if (v.size() != 4) {
    return std::nullopt;
  }
  unsigned int key =
      std::byteswap(*reinterpret_cast<unsigned int const *>(v.data()));

  return ReadKeyInfo(key).and_then([&](KeyInfo &&key_info) {
    return Read(KeyData{5, key, std::move(key_info)});
  });
}

[[nodiscard]] std::optional<KeyData>
SMC::Read(KeyData const &input) const noexcept {
  KeyData output;
  size_t len{sizeof(KeyData)};
  kern_return_t res = IOConnectCallStructMethod(conn, 2, &input,
                                                sizeof(KeyData), &output, &len);
  if (res != 0 or output.result == 132 or output.result != 0) {
    return std::nullopt;
  } else {
    return output;
  }
}

std::optional<KeyInfo> SMC::ReadKeyInfo(Key key) {
  if (auto i = map.find(key); i != map.end()) {
    return i->second;
  }
  return Read(KeyData{9, key}).transform([&](KeyData &&d) {
    return map.emplace(key, d.key_info).first->second;
  });
}

KeyData::KeyData(unsigned char data8, Key key, KeyInfo key_info) noexcept
    : key{std::move(key)}, key_info{std::move(key_info)},
      data8{std::move(data8)} {
  std::memset(bytes, 0, sizeof(bytes));
}

KeyData::KeyData(unsigned char data8, Key key) noexcept
    : KeyData{std::move(data8), std::move(key), {}} {}

KeyData::KeyData() noexcept : KeyData({}, {}) {}