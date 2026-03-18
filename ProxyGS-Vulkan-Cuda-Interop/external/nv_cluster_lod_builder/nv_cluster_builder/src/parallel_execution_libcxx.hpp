/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */
/// @file Shim for missing libc++ features. libc++ is the LLVM implementation of
/// the standard library. This project was developed with libstdc++ (the GNU
/// implementation) and MSVC STL. The contents of this file provides workarounds
/// for missing features, and disables parallel execution in the process.
/// See: https://github.com/nvpro-samples/nv_cluster_lod_builder/issues/1
/// TODO: parallel execution with e.g. https://github.com/mikekazakov/pstld
#pragma once

#include <numeric>
#include <ranges>

// TODO: add a numerical comparison for _LIBCPP_VERSION if std::execution
// support is added
#if defined(_LIBCPP_VERSION)

// Disable parallel execution as it is not supported by libc++ or this shim
#if !defined(NVCLUSTER_MULTITHREADED)
#define NVCLUSTER_MULTITHREADED 0
#else
#undef NVCLUSTER_MULTITHREADED
#define NVCLUSTER_MULTITHREADED 0
#endif

namespace std {

// If you see duplicate definitions here, filter out the current _LIBCPP_VERSION
namespace execution {
class sequenced_policy
{
};
class parallel_policy
{
};
class parallel_unsequenced_policy
{
};
class unsequenced_policy
{
};
inline constexpr sequenced_policy            seq{};
inline constexpr parallel_policy             par{};
inline constexpr parallel_unsequenced_policy par_unseq{};
inline constexpr unsequenced_policy          unseq{};
}  // namespace execution

template <class ExecutionPolicy, class ForwardIt, class UnaryFunc>
void for_each(ExecutionPolicy&&,

              ForwardIt first,
              ForwardIt last,
              UnaryFunc f)
{
  for_each(first, last, f);
}

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2>
  requires std::same_as<std::decay_t<ExecutionPolicy>, execution::sequenced_policy>
ForwardIt2 inclusive_scan(ExecutionPolicy&&, ForwardIt1 first, ForwardIt1 last, ForwardIt2 d_first)
{
  return inclusive_scan(first, last, d_first);
}

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2, class T>
  requires std::same_as<std::decay_t<ExecutionPolicy>, execution::sequenced_policy>
           || std::same_as<std::decay_t<ExecutionPolicy>, execution::parallel_policy>
           || std::same_as<std::decay_t<ExecutionPolicy>, execution::parallel_unsequenced_policy>
           || std::same_as<std::decay_t<ExecutionPolicy>, execution::unsequenced_policy>
ForwardIt2 exclusive_scan(ExecutionPolicy&&, ForwardIt1 first, ForwardIt1 last, ForwardIt2 d_first, T init)
{
  static_assert(std::same_as<std::decay_t<ExecutionPolicy>, execution::sequenced_policy>);  // SFINAE delayed error
  return exclusive_scan(first, last, d_first, init);
}

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2, class T, class BinaryOp>
  requires std::same_as<std::decay_t<ExecutionPolicy>, execution::sequenced_policy>
ForwardIt2 exclusive_scan(ExecutionPolicy&&, ForwardIt1 first, ForwardIt1 last, ForwardIt2 d_first, T init, BinaryOp op)
{
  return exclusive_scan(first, last, d_first, init, op);
}

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2, class T, class BinaryOp, class UnaryOp>
  requires std::same_as<std::decay_t<ExecutionPolicy>, execution::sequenced_policy>
ForwardIt2 transform_exclusive_scan(ExecutionPolicy&&, ForwardIt1 first, ForwardIt1 last, ForwardIt2 d_first, T init, BinaryOp binary_op, UnaryOp unary_op)
{
  return transform_exclusive_scan(first, last, d_first, init, binary_op, unary_op);
}

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2, class BinaryOp, class UnaryOp>
  requires std::same_as<std::decay_t<ExecutionPolicy>, execution::sequenced_policy>
ForwardIt2 transform_inclusive_scan(ExecutionPolicy&&,
                                    ForwardIt1 first,
                                    ForwardIt1 last,
                                    ForwardIt2 d_first,

                                    BinaryOp binary_op,
                                    UnaryOp  unary_op)
{
#if 1
  auto transformed_view = std::ranges::subrange(first, last) | std::views::transform(unary_op);
  return std::inclusive_scan(transformed_view.begin(), transformed_view.end(), d_first, binary_op);
#else
  // possible bug in libc++: typename iterator_traits<_InputIterator>::value_type __init = __u(*__first);
  return transform_inclusive_scan(first, last, d_first, binary_op, unary_op);
#endif
}

template <class ExecutionPolicy, class BidirIt, class UnaryPred>
  requires std::same_as<std::decay_t<ExecutionPolicy>, execution::sequenced_policy>
BidirIt stable_partition(ExecutionPolicy&&,

                         BidirIt   first,
                         BidirIt   last,
                         UnaryPred p)
{
  return stable_partition(first, last, p);
}

template <class ExecutionPolicy, class RandomIt>
  requires std::same_as<std::decay_t<ExecutionPolicy>, execution::sequenced_policy>
void sort(ExecutionPolicy&&, RandomIt first, RandomIt last)
{
  return sort(first, last);
}

template <class ExecutionPolicy, class RandomIt, class Compare>
  requires std::same_as<std::decay_t<ExecutionPolicy>, execution::sequenced_policy>
void sort(ExecutionPolicy&&, RandomIt first, RandomIt last, Compare comp)
{
  return sort(first, last, comp);
}

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2, class T>
  requires std::same_as<std::decay_t<ExecutionPolicy>, execution::sequenced_policy>
T transform_reduce(ExecutionPolicy&&, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, T init)
{
  return transform_reduce(first1, last1, first2, init);
}

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2, class T, class BinaryOp1, class BinaryOp2>
  requires std::same_as<std::decay_t<ExecutionPolicy>, execution::sequenced_policy>
T transform_reduce(ExecutionPolicy&&, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, T init, BinaryOp1 reduce, BinaryOp2 transform)
{
  return transform_reduce(first1, last1, first2, init, reduce, transform);
}

// Workaround for missing atomic_ref in libc++
#if _LIBCPP_VERSION < 190000
struct atomic_ref
{
  atomic_ref(uint32_t& v)
      : value(&v)
  {
  }
  uint32_t  operator++() { return reinterpret_cast<std::atomic<uint32_t>&>(*value).operator++(); }
  uint32_t  operator++(int) { return reinterpret_cast<std::atomic<uint32_t>&>(*value).operator++(0); }
  uint32_t* value;
};
#endif

}  // namespace std
#endif
