#pragma once
#include <cstddef>

// SIMD
constexpr std::size_t ALIGN_SIMD128 = 16;
constexpr std::size_t ALIGN_SIMD256 = 32;

// CACHE LINE
constexpr std::size_t ALIGN_CACHE_LINE = 64;

// VECTORS
constexpr std::size_t ALIGN_VEC4F = 16;
constexpr std::size_t ALIGN_VEC3F = 4;
constexpr std::size_t ALIGN_VEC2F = 8;

// QUATERNION
constexpr std::size_t ALIGN_QUATERNION = 16;

// MATRIX
constexpr std::size_t ALIGN_MAT4X4F = 64;