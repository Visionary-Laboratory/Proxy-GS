/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <algorithm>
#include <array>
#include <assert.h>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <nvcluster/nvcluster.h>
#include <ranges>
#include <vector>

#ifdef min
#error "Preprocessor min defined. Add NOMINMAX to the build system"
#endif

#ifdef max
#undef "Preprocessor max defined. Add NOMINMAX to the build system"
#endif

namespace nvcluster {

// Returns the ceiling of an integer division. Assumes positive values.
template <std::integral T>
T div_ceil(const T& a, const T& b)
{
  return (a + b - 1) / b;
}

// A tiny and general vector implementation, like glm
// clang-format off
template<class T, std::size_t N>
  requires std::is_arithmetic_v<T>
struct vec : std::array<T, N> {
    using std::array<T, N>::array;
    using std::array<T, N>::operator[];
    using std::array<T, N>::begin;
    using std::array<T, N>::end;

    [[nodiscard]] constexpr vec() noexcept : std::array<T, N>{} {} // zero initialize
    [[nodiscard]] constexpr vec(T all) noexcept { std::ranges::fill(*this, all); }

    // Workaround for aggregate std::array initialization,
    // https://stackoverflow.com/questions/8192185
    // TODO: remove unsafe static_cast! Not sure what to do to mimic brace initialization
    template<typename... U>
      requires (sizeof...(U) == N) && (std::is_convertible_v<U, T> && ...)
    [[nodiscard]] constexpr vec(const U&... init) noexcept : std::array<T, N>{ {static_cast<T>(init)...} } {}

    // Creating an apply(..., std::plus<T>()) could work too
    constexpr vec& operator+=(const vec& v) { for (std::size_t i = 0; i < N; ++i) (*this)[i] += v[i]; return *this; }
    constexpr vec& operator-=(const vec& v) { for (std::size_t i = 0; i < N; ++i) (*this)[i] -= v[i]; return *this; }
    constexpr vec& operator*=(T s) { for (std::size_t i = 0; i < N; ++i) (*this)[i] *= s; return *this; }
    constexpr vec& operator/=(T s) { for (std::size_t i = 0; i < N; ++i) (*this)[i] /= s; return *this; }

    [[nodiscard]] constexpr vec operator-() const { vec r; for (std::size_t i = 0; i < N; ++i) r[i] = -(*this)[i]; return r; }

    // "Hidden friends" for faster compilation
    [[nodiscard]] friend constexpr vec operator+(const vec& a, const vec& b) { return vec(a) += b; }
    [[nodiscard]] friend constexpr vec operator-(const vec& a, const vec& b) { return vec(a) -= b; }
    [[nodiscard]] friend constexpr vec operator*(const vec& v, T s) { return vec(v) *= s; }
    [[nodiscard]] friend constexpr vec operator*(T s, const vec& v) { return v * s; }
    [[nodiscard]] friend constexpr vec operator/(const vec& v, T s) { return vec(v) /= s; }
    [[nodiscard]] friend constexpr bool operator==(const vec& a, const vec& b) { return std::ranges::equal(a, b); }
    [[nodiscard]] friend constexpr bool operator!=(const vec& a, const vec& b) { return !(a == b); }
};
template<class T, std::size_t N> [[nodiscard]] constexpr vec<T, N> min(const vec<T, N>& a, const vec<T, N>& b) { vec<T, N> r; for (std::size_t i = 0; i < N; ++i) r[i] = std::min(a[i], b[i]); return r; }
template<class T, std::size_t N> [[nodiscard]] constexpr vec<T, N> max(const vec<T, N>& a, const vec<T, N>& b) { vec<T, N> r; for (std::size_t i = 0; i < N; ++i) r[i] = std::max(a[i], b[i]); return r; }
template<class T, std::size_t N> [[nodiscard]] constexpr vec<T, N> clamp(const vec<T, N>& v, const vec<T, N>& min_v, const vec<T, N>& max_v) { vec<T, N> r; for (std::size_t i = 0; i < N; ++i) r[i] = std::clamp(v[i], min_v[i], max_v[i]); return r; }
template<class T, std::size_t N> [[nodiscard]] constexpr T dot(const vec<T, N>& a, const vec<T, N>& b) { T r{}; for (std::size_t i = 0; i < N; ++i) r += a[i] * b[i]; return r; }
template<class T, std::size_t N> [[nodiscard]] constexpr T length_squared(const vec<T, N>& v) { return dot(v, v); }
template<class T, std::size_t N> [[nodiscard]] T length(const vec<T, N>& v) requires std::floating_point<T> { return std::sqrt(length_squared(v)); }
template<class T, std::size_t N> [[nodiscard]] vec<T, N> normalize(const vec<T, N>& v) requires std::floating_point<T> { return v * (T{1} / length(v)); }
template<class T, std::size_t N> [[nodiscard]] constexpr vec<T, N> cross(const vec<T, N>& a, const vec<T, N>& b) requires (N == 3) && std::is_signed_v<T> {
    return {a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]};
}
// clang-format on

using vec2f = vec<float, 2>;
using vec3f = vec<float, 3>;
using vec4f = vec<float, 4>;
using vec2u = vec<uint32_t, 2>;
using vec3u = vec<uint32_t, 3>;
using vec4u = vec<uint32_t, 4>;
using vec2i = vec<int32_t, 2>;
using vec3i = vec<int32_t, 3>;
using vec4i = vec<int32_t, 4>;
static_assert(sizeof(nvcluster_Vec3f) == sizeof(vec3f));

// Axis aligned bounding box
struct AABB
{
  vec3f min, max;

  // Plus returns the union of bounding boxes.
  // [[nodiscard]] allows the compiler to warn if the return value is ignored,
  // which would be a bug. E.g. a + b; but should be a += b;
  [[nodiscard]] constexpr AABB operator+(const AABB& other) const
  {
    return {nvcluster::min(min, other.min), nvcluster::max(max, other.max)};
  }
  constexpr AABB& operator+=(const AABB& other) { return *this = *this + other; };

  [[nodiscard]] constexpr vec3f size() const { return max - min; }
  [[nodiscard]] constexpr vec3f center() const { return (min + max) * 0.5f; }
  [[nodiscard]] constexpr vec3f positive_size() const { return nvcluster::max(vec3f(0.0f), size()); }
  [[nodiscard]] constexpr AABB  positive() const { return {min, min + positive_size()}; }
  [[nodiscard]] constexpr float half_area() const
  {
    auto s = size();
    return s[0] * (s[1] + s[2]) + s[1] * s[2];
  }
  [[nodiscard]] constexpr AABB intersect(const AABB& other) const
  {
    return AABB{nvcluster::max(min, other.min), nvcluster::min(max, other.max)}.positive();
  }
  [[nodiscard]] constexpr static AABB empty()
  {
    return {vec3f{std::numeric_limits<float>::max()}, vec3f{std::numeric_limits<float>::lowest()}};
  }
  operator nvcluster_AABB() const { return reinterpret_cast<const nvcluster_AABB&>(*this); }
};
static_assert(sizeof(nvcluster_AABB) == sizeof(AABB));

// An index/cursor based subrange
struct Range
{
  uint32_t offset = {};
  uint32_t count  = {};

  // Use iota() to make the range iterable
  // E.g.: for(uint32_t i : range.indices()) ...
  // std::views::iota() is similar to python's range()
  [[nodiscard]] auto               indices() const { return std::views::iota(offset, offset + count); }
  [[nodiscard]] constexpr uint32_t end() const { return offset + count; }
  operator nvcluster_Range() { return {offset, count}; }
};
static_assert(sizeof(nvcluster_Range) == sizeof(Range));

}  // namespace nvcluster

// hashing functions from https://stackoverflow.com/questions/35985960/c-why-is-boosthash-combine-the-best-way-to-combine-hash-values
namespace {

template <typename T>
constexpr T xorshift(const T& n, int i)
{
  return n ^ (n >> i);
}

inline constexpr uint32_t hash(const uint32_t& n)
{
  uint32_t p = 0x55555555ul;  // pattern of alternating 0 and 1
  uint32_t c = 3423571495ul;  // random uneven integer constant;
  return c * xorshift(p * xorshift(n, 16), 16);
}

inline constexpr uint64_t hash(const uint64_t& n)
{
  uint64_t p = 0x5555555555555555ull;    // pattern of alternating 0 and 1
  uint64_t c = 17316035218449499591ull;  // random uneven integer constant;
  return c * xorshift(p * xorshift(n, 32), 32);
}

// call this function with the old seed and the new key to be hashed and
// combined into the new seed value, respectively the final hash
template <class T>
constexpr size_t hash_combine(std::size_t& seed, const T& v)
{
  return seed = std::rotl(seed, std::numeric_limits<size_t>::digits / 3) ^ hash(std::hash<T>{}(v));
}

// From: https://blog.infotraining.pl/how-to-hash-objects-without-repetition
template <class... TValues>
  requires(sizeof...(TValues) > 1)
constexpr size_t combined_hash(const TValues&... values)
{
  size_t seed{};
  (..., hash_combine(seed, values));
  return seed;
}

// Adapter for std::array
template <class T, size_t N>
constexpr size_t array_hash(const std::array<T, N>& arr)
{
  return [&arr]<std::size_t... I>(std::index_sequence<I...>) { return combined_hash(arr[I]...); }(std::make_index_sequence<N>{});
}

}  // anonymous namespace

// Define a hash so vec3 can be used in e.g. std::unordered_map
template <class T, size_t N>
struct std::hash<nvcluster::vec<T, N>>
{
  std::size_t operator()(const nvcluster::vec<T, N>& v) const noexcept { return array_hash<T, N>(v); }
};
