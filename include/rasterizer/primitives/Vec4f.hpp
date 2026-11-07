#pragma once
#include <cmath>
#include <cstddef>
#include <cassert>
#include "rasterizer/math_core/Align.hpp"
#include "rasterizer/primitives/VecConfig.hpp"
#include "rasterizer/primitives/Vec2f.hpp"
#include "rasterizer/primitives/Vec3f.hpp"
#include <rasterizer/simd/SimdMath.hpp>

union Vec3f;
union Vec2f;

// Represents a 16-byte aligned 4-Dimensional Vector backed by a 128-bit SIMD register
union alignas(ALIGN_VEC4F) Vec4f
{
    struct {float x, y, z, w;};
    struct {float arrView[4];};
    __m128 simdView;

#// region Constructors
    // Initializes all components to zero
    inline Vec4f() {
        this->simdView = _mm_setzero_ps();
    }

    // Initializes directly from an existing 128-bit SIMD register
    inline Vec4f(__m128 simd) {
        this->simdView = simd;
    }

    // Initializes components in little-endian lane order: [x, y, z, w]
    inline Vec4f(float x, float y, float z, float w) {
        this->simdView = _mm_setr_ps(x, y, z, w);
    }

    // Default copy constructor
    inline Vec4f(const Vec4f& other) = default;

    // Broadcasts a single scalar value to all 4 SIMD lanes
    inline Vec4f(float v) {
        this->simdView = _mm_set1_ps(v);
    }
    
    // Initializes components from a 4-element float array in order
    explicit inline Vec4f(float values[4]) {
        this->simdView = _mm_setr_ps(values[0], values[1], values[2], values[3]);
    }

    // Loads 4 contiguous floats from a 16-byte aligned memory address
    explicit inline Vec4f(const float *ptr) {
        this->simdView = _mm_load_ps(ptr);
    }
//#endregion

#// region AssignOperator
    // Default copy assignment operator
    inline Vec4f &operator=(const Vec4f& other) = default;
//#endregion

#// region ArithmeticalOperators
    // Component-wise vector addition
    inline Vec4f MATH_VECTORCALL operator+(const Vec4f& other) const {
        Vec4f res = *this;
        return res += other;
    }
    
    // In-place component-wise vector addition
    inline const Vec4f& MATH_VECTORCALL operator+=(const Vec4f& other) {
        this->simdView = _mm_add_ps(this->simdView, other.simdView);
        return *this;
    }

    // Component-wise vector subtraction
    inline Vec4f MATH_VECTORCALL operator-(const Vec4f& other) const {
        Vec4f res = *this;
        return res -= other;
    }

    // Unary negation (flips the sign of all components)
    inline Vec4f MATH_VECTORCALL operator-() const {
        // Flips the IEEE-754 sign bit (MSB) of each lane via bitwise XOR with -0.0f (0x80000000)
        return _mm_xor_ps(this->simdView, _mm_set1_ps(-0.0f));
    }

    // In-place component-wise vector subtraction
    inline const Vec4f& MATH_VECTORCALL operator-=(const Vec4f& other) {
        this->simdView = _mm_sub_ps(this->simdView, other.simdView);
        return *this;
    }

    // Component-wise multiplication (Hadamard product)
    inline Vec4f MATH_VECTORCALL operator*(const Vec4f& other) const {
        Vec4f res = *this;
        return res *= other;
    }

    // In-place component-wise multiplication (Hadamard product)
    inline const Vec4f& MATH_VECTORCALL operator*=(const Vec4f& other) {
        this->simdView = _mm_mul_ps(this->simdView, other.simdView);
        return *this;
    }

    // Multiplies all components by a scalar
    inline Vec4f MATH_VECTORCALL operator*(float scalar) const {
        Vec4f res = *this;
        return res *= scalar;
    }

    // In-place multiplication by a scalar
    inline const Vec4f& MATH_VECTORCALL operator*=(float scalar) {
        this->simdView = _mm_mul_ps(this->simdView, _mm_set1_ps(scalar));
        return *this;
    }

    // Divides all components by a scalar
    inline Vec4f MATH_VECTORCALL operator/(float scalar) const {
        Vec4f res = *this;
        return res /= scalar;
    }
    
    // In-place division by a scalar
    inline const Vec4f& MATH_VECTORCALL operator/=(float scalar) {
        // Precomputes reciprocal to replace slow SIMD division with multiplication
        scalar = 1.0f / scalar;
        this->simdView = _mm_mul_ps(this->simdView, _mm_set1_ps(scalar));
        return *this;
    }

    // Computes Fused Multiply-Add: (a * b) + c with a single rounding step
    inline Vec4f MATH_VECTORCALL fmadd(const Vec4f& a, const Vec4f& b, const Vec4f& c) const {
        return _mm_fmadd_ps(a.simdView, b.simdView, c.simdView);
    }

    // In-place Fused Multiply-Add: accumulates (a * b) into this vector
    inline Vec4f& MATH_VECTORCALL fmadd(const Vec4f& a, const Vec4f& b) {
        this->simdView = _mm_fmadd_ps(a.simdView, b.simdView, this->simdView);
        return *this;
    }

    // Computes Fused Multiply-Subtract: (a * b) - c with a single rounding step
    inline Vec4f MATH_VECTORCALL fmsub(const Vec4f& a, const Vec4f& b, const Vec4f& c) const {
        return _mm_fmsub_ps(a.simdView, b.simdView, c.simdView);
    }

    // In-place Fused Multiply-Subtract: computes (a * b) - this and stores it in this vector
    inline Vec4f& MATH_VECTORCALL fmsub(const Vec4f& a, const Vec4f& b) {
        this->simdView = _mm_fmsub_ps(a.simdView, b.simdView, this->simdView);
        return *this;
    }
//#endregion

#// region LogicalOperators
    // Exact component-wise equality check
    inline bool MATH_VECTORCALL operator==(const Vec4f& other) const {
        // Extracts the 4 comparison MSBs into a 4-bit integer mask (0x0F means all 4 lanes match)
        return _mm_movemask_ps(_mm_cmpeq_ps(this->simdView, other.simdView)) == 0x0F;
    }

    // Exact component-wise inequality check
    inline bool MATH_VECTORCALL operator!=(const Vec4f& other) const {
        return _mm_movemask_ps(_mm_cmpeq_ps(this->simdView, other.simdView)) != 0x0F;
    }

    // Checks if all components are equal within a given epsilon tolerance
    inline bool IsNearlyEqual(const Vec4f& other, float epsilon = 1e-4f) const {
        // Clears the sign bit of (this - other) via ANDNOT with -0.0f to compute fast absolute difference
        __m128 absDiff = _mm_andnot_ps(_mm_set1_ps(-0.0f), _mm_sub_ps(this->simdView, other.simdView));
        // Checks if all 4 lanes are <= epsilon
        return _mm_movemask_ps(_mm_cmple_ps(absDiff, _mm_set1_ps(epsilon))) == 0x0F;
    }
//#endregion

#// region IndexingOperators
    // Returns component at index (read-only)
    inline float MATH_VECTORCALL operator[](const size_t index) const {
        assert(index < 4);
        return this->arrView[index];
    }

    // Returns mutable reference to component at index
    inline float& MATH_VECTORCALL operator[](const size_t index) {
        assert(index < 4);
        return this->arrView[index];
    }
//#endregion

#// region VectorOperators
    // Computes 4D dot product (x*x + y*y + z*z + w*w)
    inline float MATH_VECTORCALL Dot(const Vec4f& other) const {
        // Mask 0xFF: multiplies all 4 lanes (upper nibble 0xF) and broadcasts result to all lanes (lower nibble 0xF)
        return _mm_cvtss_f32(_mm_dp_ps(this->simdView, other.simdView, 0xFF));
    }

    // Computes 3D dot product ignoring the w channel (x*x + y*y + z*z)
    inline float MATH_VECTORCALL Dot3(const Vec4f& other) const {
        // Mask 0x7F: multiplies only x, y, z lanes (upper nibble 0x7 = 0111b) and broadcasts to all lanes (0xF)
        return _mm_cvtss_f32(_mm_dp_ps(this->simdView, other.simdView, 0x7F));
    }

    // Computes squared magnitude of the vector
    inline float MATH_VECTORCALL SquaredMagnitude() const {
        return this->Dot(*this);
    }

    // Computes Euclidean magnitude of the vector
    inline float MATH_VECTORCALL Magnitude() const {
        return std::sqrt(this->SquaredMagnitude());
    }

    // In-place fast vector normalization using hardware reciprocal square root approximation
    inline Vec4f& MATH_VECTORCALL Normalize() {
        // Computes squared length x = Dot(v, v) broadcasted across all lanes
        __m128 squaredLength = _mm_dp_ps(simdView, simdView, 0xFF);
        // Fast hardware approximation of 1 / sqrt(x) (~12-bit precision)
        __m128 y0 = _mm_rsqrt_ps(squaredLength);

        // One Newton-Raphson iteration restores full ~23-bit float precision:
        // y1 = -0.5 * y0 * (x * y0^2 - 3.0)
        const __m128 minus_half  = _mm_set1_ps(-0.5f);
        const __m128 minus_three = _mm_set1_ps(-3.0f);

        __m128 y0_sq         = _mm_mul_ps(y0, y0);
        __m128 minus_half_y0 = _mm_mul_ps(minus_half, y0);

        // Computes (x * y0^2 - 3.0) via single-cycle FMA
        __m128 sub = _mm_fmadd_ps(squaredLength, y0_sq, minus_three);

        // Guards against zero-length vector NaNs by masking out lanes where squaredMagnitude ~ 0
        __m128 validMask = _mm_cmpgt_ps(squaredLength, _mm_set1_ps(1e-8f));
        __m128 invLen    = _mm_and_ps(_mm_mul_ps(minus_half_y0, sub), validMask);

        simdView = _mm_mul_ps(simdView, invLen);

        return *this;
    }

    // Returns a normalized copy of this vector using fast hardware rsqrt
    inline Vec4f MATH_VECTORCALL Normalize() const {
        Vec4f res = *this;
        res.Normalize();
        return res;
    }
//#endregion

#// region ConversionOperators
    // Truncates z and w to return a 2D vector (x, y)
    inline Vec2f ToVec2f() const;

    // Truncates w to return a 3D vector (x, y, z)
    inline Vec3f ToVec3f() const;
//#endregion
};