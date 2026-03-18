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

#include <execution>

// Workaround for libc++ std::execution
#include <parallel_execution_libcxx.hpp>

// Shortcut to select the parallel execution model depending on a bool, using
// variable template specialization
template <bool Parallelize>
inline constexpr auto exec = std::execution::seq;
template <>
inline constexpr auto exec<true> = std::execution::par_unseq;

// This is an iterator that counts upwards from an initial value.
// std::views::iota would almost work for this, but iota on MSVC 2019 doesn't
// support random access, which is necessary for parallelism.
template <class T>
struct iota_iterator
{
  using value_type = T;
  // [iterator.traits] in the C++ standard requires this to be a signed type.
  // We choose int64_t here, because it's conceivable someone could use
  // T == uint32_t and then iterate over more than 2^31 - 1 elements.
  using difference_type                                         = int64_t;
  using pointer                                                 = T*;
  using reference                                               = T&;
  using iterator_category                                       = std::random_access_iterator_tag;
  iota_iterator()                                               = default;
  iota_iterator(const iota_iterator& other) noexcept            = default;
  iota_iterator(iota_iterator&& other) noexcept                 = default;
  iota_iterator& operator=(const iota_iterator& other) noexcept = default;
  iota_iterator& operator=(iota_iterator&& other) noexcept      = default;
  iota_iterator(T i_)
      : i(i_)
  {
  }
  value_type     operator*() const { return i; }
  iota_iterator& operator++()
  {
    ++i;
    return *this;
  }
  iota_iterator operator++(int)
  {
    iota_iterator t(*this);
    ++*this;
    return t;
  }
  iota_iterator& operator--()
  {
    --i;
    return *this;
  }
  iota_iterator operator--(int)
  {
    iota_iterator t(*this);
    --*this;
    return t;
  }
  iota_iterator  operator+(difference_type d) const { return {static_cast<T>(static_cast<difference_type>(i) + d)}; }
  iota_iterator  operator-(difference_type d) const { return {static_cast<T>(static_cast<difference_type>(i) - d)}; }
  iota_iterator& operator+=(difference_type d)
  {
    i = static_cast<T>(static_cast<difference_type>(i) + d);
    return *this;
  }
  iota_iterator& operator-=(difference_type d)
  {
    i = static_cast<T>(static_cast<difference_type>(i) - d);
    return *this;
  }
  bool                 operator==(const iota_iterator& other) const { return i == other.i; }
  bool                 operator!=(const iota_iterator& other) const { return i != other.i; }
  bool                 operator<(const iota_iterator& other) const { return i < other.i; }
  bool                 operator<=(const iota_iterator& other) const { return i <= other.i; }
  bool                 operator>(const iota_iterator& other) const { return i > other.i; }
  bool                 operator>=(const iota_iterator& other) const { return i >= other.i; }
  difference_type      operator-(const iota_iterator& other) const
  {
    return static_cast<difference_type>(i) - static_cast<difference_type>(other.i);
  }
  friend iota_iterator operator+(difference_type n, const iota_iterator& it) { return it + n; }
  T operator[](difference_type d) const { return static_cast<T>(static_cast<difference_type>(i) + d); }

private:
  T i = 0;
};

// Expresses the range from m_begin to m_end - 1.
template <class T>
struct iota_view
{
  using iterator = iota_iterator<T>;
  iota_view(T begin, T end)
      : m_begin(begin)
      , m_end(end)
  {
  }
  iterator begin() const { return {m_begin}; };
  iterator end() const { return {m_end}; };

private:
  T m_begin, m_end;
};

// Runs a function in parallel for each index from 0 to numItems - 1. Uses
// batches of size BatchSize for reduced overhead and better autovectorization.
//
// BatchSize will also be used as the threshold for when to switch from
// single-threaded to multi-threaded execution. For this reason, it should be set
// to a power of 2 around where multi - threaded is faster than single - threaded for
// the given function.Some examples are :
// * 8192 for trivial workloads(a * x + y)
// * 2048 for animation workloads(multiplication by a single matrix)
// * 512 for more computationally heavy workloads(run XTEA)
// * 1 for full parallelization(load an image)
//
// This is a simpler version of nvh::parallel_batches, which you can find in
// nvpro_core.
template <bool Parallelize = true, size_t BatchSize = 512, typename F>
inline void parallel_batches(size_t numItems, F&& fn)
{
  if constexpr(!Parallelize)
  {
    // Explicit constexpr case to avoid linking to the parallel implementation
    // if it's not used (and can't partially specialize the function).
    for(size_t i = 0; i < numItems; i++)
    {
      fn(i);
    }
  }
  else
  {
    // For small item counts, it's fastest to use a single thread and avoid the
    // overhead from invoking a parallel executor.
    if(numItems <= BatchSize)
    {
      for(size_t i = 0; i < numItems; i++)
      {
        fn(i);
      }
    }
    else
    {
      // Unroll the loop into batches of size BATCHSIZE or less. This worker
      // function will be run in parallel using
      // std::for_each(std::execution::par_unseq).
      const size_t numBatches = (numItems + BatchSize - 1) / BatchSize;
      auto         worker     = [&numItems, &fn](const size_t batchIndex) {
        const size_t start          = BatchSize * batchIndex;
        const size_t itemsRemaining = numItems - start;
        // This split is necessary to make MSVC try to auto-vectorize the first
        // loop, which will be the most common case when numItems is large.
        if(itemsRemaining >= BatchSize)
        {
          // Exactly BATCHSIZE items to process
          for(size_t i = start; i < start + BatchSize; i++)
          {
            fn(i);
          }
        }
        else
        {
          // Variable-length loop
          for(size_t i = start; i < numItems; i++)
          {
            fn(i);
          }
        }
      };

      // This runs the worker above for each batch from 0 to numBatches-1.
      iota_view<size_t> batches(0, numBatches);
      std::for_each(std::execution::par_unseq, batches.begin(), batches.end(), worker);
    }
  }
}
