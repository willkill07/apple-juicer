#pragma once

#include <array>
#include <bit>
#include <string_view>

template <auto V> struct Value {};

template <auto V, auto... Vs>
  requires(... and std::same_as<decltype(V), decltype(Vs)>)
struct Values {};

using std::string_view_literals::operator""sv;

template <typename T>
constexpr std::string_view kLabel{std::array{
    "fp8"sv, "fp16"sv, "fp32"sv, "fp64"sv}[std::countr_zero(sizeof(T))]};
