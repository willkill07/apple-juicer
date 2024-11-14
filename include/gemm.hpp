#pragma once

#include <algorithm>
#include <memory>
#include <random>

#define ACCELERATE_NEW_LAPACK 1
#define ACCELERATE_LAPACK_ILP64
#include <Accelerate/Accelerate.h>
#include <vecLib/vecLib.h>

constexpr std::size_t kNumMatrices{3};

constexpr std::array kBases{2, 3, 5, 6, 7, 10};

template <std::ranges::random_access_range Rng>
consteval bool ValidateBases(Rng const &rng) noexcept {
  return std::ranges::none_of(rng, [&](auto const &v) { //
    return std::ranges::contains(rng, v * v);
  });
}

static_assert(ValidateBases(kBases));

template <typename Type> //
struct GemmData {
  std::size_t elems;
  std::unique_ptr<Type[]> a;
  std::unique_ptr<Type[]> b;
  std::unique_ptr<Type[]> c;
  Type alpha;
  Type beta;

  GemmData(std::size_t max_n)
      : elems{max_n * max_n}, //
        a{std::make_unique_for_overwrite<Type[]>(elems)},
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
  std::size_t const max_elems{maximum / (kNumMatrices * sizeof(T))};
  std::vector<std::size_t> res;
  for (std::size_t base : kBases) {
    for (std::size_t dim{base}; dim * dim <= max_elems; dim *= base) {
      res.push_back(dim);
    }
  }
  std::ranges::sort(res);
  return res;
}
