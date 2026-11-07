#pragma once
#include <cassert>
#include <cstdlib>
#include <cmath>
#include "rasterizer/primitives/VecConfig.hpp"
#include "rasterizer/primitives/Vec2f.hpp"
#include <rasterizer/simd/SimdMath.hpp>

union Vec4f;
union Vec2f;

// Represents a tightly-packed 12-byte 3-Dimensional Vector (spatial XYZ or color RGB)
union Vec3f
{
    struct {float x, y, z;};
    struct {float r, g, b;};
    float arrView[3];

//#region Constructors
    // Initializes all components to zero
    inline Vec3f() {
        this->x = 0;
        this->y = 0;
        this->z = 0;
    }

    // Initializes components from individual x, y, z scalars
    inline Vec3f(float x, float y, float z) {
        this->x = x;
        this->y = y;
        this->z = z;
    }

    // Default copy constructor
    inline Vec3f(const Vec3f& other) = default;

    // Broadcasts a single scalar value to all 3 components
    explicit inline Vec3f(float v) {
        this->x = v;
        this->y = v;
        this->z = v;
    }
    
    // Initializes components from a 3-element float array
    explicit inline Vec3f(float values[3]) {
        this->x = values[0];
        this->y = values[1];
        this->z = values[2];
    }

    // Loads 3 contiguous floats from a memory pointer
    explicit inline Vec3f(const float *ptr) {
        this->x = ptr[0];
        this->y = ptr[1];
        this->z = ptr[2];
    }
//#endregion

//#region AssignOperator
    // Default copy assignment operator
    inline Vec3f& operator=(const Vec3f& other) = default;
//#endregion

//#region ArithmeticalOperators
    // Component-wise vector addition
    inline Vec3f operator+(const Vec3f& other) const {
        Vec3f res = *this;
        return res += other;
    }
    
    // In-place component-wise vector addition
    inline Vec3f& operator+=(const Vec3f& other) {
        this->x += other.x;
        this->y += other.y;
        this->z += other.z;
        return *this;
    }

    // Component-wise vector subtraction
    inline Vec3f operator-(const Vec3f& other) const {
        Vec3f res = *this;
        return res -= other;
    }

    // Unary negation (flips the sign of all components)
    inline Vec3f operator-() const {
        return {-this->x, -this->y, -this->z};
    }

    // In-place component-wise vector subtraction
    inline Vec3f &operator-=(const Vec3f& other) {
        this->x -= other.x;
        this->y -= other.y;
        this->z -= other.z;
        return *this;
    }

    // Component-wise multiplication (Hadamard product)
    inline Vec3f operator*(const Vec3f& other) const {
        Vec3f res = *this;
        return res *= other;
    }

    // In-place component-wise multiplication (Hadamard product)
    inline Vec3f &operator*=(const Vec3f& other) {
        this->x *= other.x;
        this->y *= other.y;
        this->z *= other.z;
        return *this;
    }

    // Multiplies all components by a scalar
    inline Vec3f operator*(float scalar) const {
        Vec3f res = *this;
        return res *= scalar;
    }

    // In-place multiplication by a scalar
    inline Vec3f &operator*=(float scalar) {
        this->x *= scalar;
        this->y *= scalar;
        this->z *= scalar;
        return *this;
    }

    // Divides all components by a scalar
    inline Vec3f operator/(float scalar) const {
        Vec3f res = *this;
        return res /= scalar;
    }

    // In-place division by a scalar
    inline Vec3f &operator/=(float scalar) {
        // Multiplies by reciprocal to avoid costly float divisions
        scalar = 1.0f / scalar;
        this->x *= scalar;
        this->y *= scalar;
        this->z *= scalar;
        return *this;
    }
//#endregion

//#region LogicalOperators
    // Exact component-wise equality check
    inline bool operator==(const Vec3f& other) const {
        return this->x == other.x && this->y == other.y && this->z == other.z;
    }

    // Exact component-wise inequality check
    inline bool operator!=(const Vec3f& other) const {
        return !(*this == other);
    }

    // Checks if all components are equal within a given epsilon tolerance
    inline bool IsNearlyEqual(const Vec3f& other, float epsilon = 1e-4f) const {
        Vec3f diff = (*this - other);
        return std::abs(diff.x) < epsilon && std::abs(diff.y) < epsilon && std::abs(diff.z) < epsilon;
    }
//#endregion

//#region IndexingOperators
    // Returns component at index (read-only)
    inline float operator[](const size_t index) const {
        assert(index < 3);
        return this->arrView[index];
    }

    // Returns mutable reference to component at index
    inline float &operator[](const size_t index) {
        assert(index < 3);
        return this->arrView[index];
    }
//#endregion

//#region VectorOperators
    // Computes 3D dot product using hardware SIMD dot-product instruction
    inline float MATH_VECTORCALL Dot(const Vec3f& other) const {
        // Packs 3D vectors into 128-bit registers with zeroed w lane
        __m128 this_simdView = _mm_setr_ps(this->x, this->y, this->z, 0);
        __m128 other_simdView = _mm_setr_ps(other.x, other.y, other.z, 0);
        // Mask 0x7F sums products of lanes 0, 1, 2 and extracts the scalar result
        return _mm_cvtss_f32(_mm_dp_ps(this_simdView, other_simdView, 0x7F));
    }

    // Computes squared magnitude of the vector
    inline float MATH_VECTORCALL SquaredMagnitude() const {
        return this->Dot(*this);
    }

    // Computes Euclidean magnitude of the vector
    inline float MATH_VECTORCALL Magnitude() const {
        return std::sqrt(this->SquaredMagnitude());
    }

    // In-place high-precision SIMD normalization using rsqrt + 1 Newton-Raphson refinement step
    inline Vec3f& MATH_VECTORCALL Normalize() {
        __m128 simdView = _mm_setr_ps(this->x, this->y, this->z, 0.0f);
        // Computes squared length x = Dot3(v, v) broadcasted across all lanes
        __m128 squaredLength = _mm_dp_ps(simdView, simdView, 0x7F);
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

        // Unpacks normalized SIMD lanes back to scalar members
        alignas(16) float temp[4];
        _mm_store_ps(temp, simdView);

        this->x = temp[0];
        this->y = temp[1];
        this->z = temp[2];
        return *this;
    }

    // Returns a normalized copy of this vector
    inline Vec3f MATH_VECTORCALL Normalized() const {
        Vec3f res = *this;
        res.Normalize();
        return res;
    }

    // Computes 3D Cross Product (this x other) using SIMD shuffles and FMA
    inline Vec3f MATH_VECTORCALL Cross(const Vec3f& other) const {
        __m128 a = _mm_setr_ps(this->x, this->y, this->z, 0);
        __m128 b = _mm_setr_ps(other.x, other.y, other.z, 0);

        // Permutes vectors for first product: a.yzx * b.zxy
        __m128 a_yzx = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1));
        __m128 b_zxy = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 1, 0, 2));
        // Permutes vectors for second product: a.zxy * b.yzx
        __m128 a_zxy = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2));
        __m128 b_yzx = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1));
        
        __m128 term2 = _mm_mul_ps(a_zxy, b_yzx);

        // Computes (a_yzx * b_zxy) - term2 in a single FMA instruction
        __m128 res = _mm_fmsub_ps(a_yzx, b_zxy, term2);
        
        alignas(16) float temp[4];
        _mm_store_ps(temp, res);

        return {temp[0], temp[1], temp[2]};
    }
//#endregion

//#region ConversionOperators
    // Truncates z to return a 2D vector (x, y)
    inline Vec2f ToVec2f() const;
    
    // Extends to a 4D vector with custom w component
    inline Vec4f ToVec4f(float w) const;

    // Extends to a 4D homogeneous direction vector (w = 0.0f, immune to translation)
    inline Vec4f ToVec4f_Direction() const;
    
    // Extends to a 4D homogeneous point in space (w = 1.0f, affected by translation)
    inline Vec4f ToVec4f_Point() const;
//#endregion
};