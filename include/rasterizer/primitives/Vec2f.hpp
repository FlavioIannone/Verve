#pragma once
#include <cassert>
#include <cmath>
#include "rasterizer/primitives/Vec4f.hpp"
#include "rasterizer/primitives/Vec3f.hpp"
#include "rasterizer/math_core/Align.hpp"

union Vec3f;
union Vec4f;

// Represents an aligned 2-Dimensional Vector (2D screen coordinates XY or texture UVs)
union alignas(ALIGN_VEC2F) Vec2f
{
    struct {float x, y;};
    struct {float u, v;};
    float arrView[2];

//#region Constructors
    // Initializes all components to zero
    inline Vec2f() {
        this->x = 0;
        this->y = 0;
    }

    // Initializes components from individual x, y scalars
    inline Vec2f(float x, float y) {
        this->x = x;
        this->y = y;
    }

    // Default copy constructor
    inline Vec2f(const Vec2f& other) = default;

    // Broadcasts a single scalar value to both components
    explicit inline Vec2f(float v) {
        this->x = v;
        this->y = v;
    }
    
    // Initializes components from a 2-element float array
    explicit inline Vec2f(float values[2]) {
        this->x = values[0];
        this->y = values[1];
    }

    // Loads 2 contiguous floats from a memory pointer
    explicit inline Vec2f(const float *ptr) {
        this->x = ptr[0];
        this->y = ptr[1];
    }
#pragma endregion

//#region AssignOperator
    // Default copy assignment operator
    inline Vec2f &operator=(const Vec2f& other) = default;
#pragma endregion

//#region ArithmeticalOperators
    // Component-wise vector addition
    inline Vec2f operator+(const Vec2f& other) const {
        Vec2f res = *this;
        return res += other;
    }
    
    // In-place component-wise vector addition
    inline Vec2f &operator+=(const Vec2f& other) {
        this->x += other.x;
        this->y += other.y;
        return *this;
    }

    // Component-wise vector subtraction
    inline Vec2f operator-(const Vec2f& other) const {
        Vec2f res = *this;
        return res -= other;
    }

    // Unary negation (flips the sign of both components)
    inline Vec2f operator-() const {
        return {-this->x, -this->y};
    }

    // In-place component-wise vector subtraction
    inline Vec2f &operator-=(const Vec2f& other) {
        this->x -= other.x;
        this->y -= other.y;
        return *this;
    }

    // Component-wise multiplication (Hadamard product)
    inline Vec2f operator*(const Vec2f& other) const {
        Vec2f res = *this;
        return res *= other;
    }

    // In-place component-wise multiplication (Hadamard product)
    inline Vec2f &operator*=(const Vec2f& other) {
        this->x *= other.x;
        this->y *= other.y;
        return *this;
    }

    // Multiplies both components by a scalar
    inline Vec2f operator*(float scalar) const {
        Vec2f res = *this;
        return res *= scalar;
    }

    // In-place multiplication by a scalar
    inline Vec2f &operator*=(float scalar) {
        this->x *= scalar;
        this->y *= scalar;
        return *this;
    }

    // Divides both components by a scalar
    inline Vec2f operator/(float scalar) const {
        Vec2f res = *this;
        return res /= scalar;
    }

    // In-place division by a scalar
    inline Vec2f &operator/=(float scalar) {
        scalar = 1.0f / scalar;
        this->x *= scalar;
        this->y *= scalar;
        return *this;
    }
#pragma endregion

//#region LogicalOperators
    // Exact component-wise equality check
    inline bool operator==(const Vec2f& other) const {
        return this->x == other.x && this->y == other.y;
    }

    // Exact component-wise inequality check
    inline bool operator!=(const Vec2f& other) const {
        return !(*this == other);
    }

    // Checks if both components are equal within a given epsilon tolerance
    inline bool IsNearlyEqual(const Vec2f& other, float epsilon = 1e-4f) const {
        Vec2f diff = (*this - other);
        return std::abs(diff.x) < epsilon && std::abs(diff.y) < epsilon;
    }
#pragma endregion

//#region IndexingOperators
    // Returns component at index (read-only)
    inline float operator[](const size_t index) const {
        assert(index < 2);
        return this->arrView[index];
    }

    // Returns mutable reference to component at index
    inline float &operator[](const size_t index) {
        assert(index < 2);
        return this->arrView[index];
    }
#pragma endregion

//#region VectorOperators
    // Computes 2D dot product (x*x + y*y)
    inline float Dot(const Vec2f& other) const {
        return this->x * other.x + this->y * other.y;
    }

    // Computes squared magnitude of the vector
    inline float SquaredMagnitude() const {
        return this->Dot(*this);
    }

    // Computes Euclidean magnitude of the vector
    inline float Magnitude() const {
        return std::sqrt(this->SquaredMagnitude());
    }

    // In-place vector normalization with zero-length safety check
    inline Vec2f& Normalize() {
        float length = this->Magnitude();

        if (length > 0.0f) {
            float invLength = 1.0f / length;
            this->x *= invLength;
            this->y *= invLength;
        }
        return *this;
    }

    // Returns a normalized copy of this vector
    inline Vec2f Normalized() const {
        Vec2f res = *this;
        return res.Normalize();
    }
#pragma endregion

//#region ConversionOperators
    // Extends to a 3D vector with optional z component
    inline Vec3f ToVec3f(float z = 0.0f) const;
    
    // Extends to a 4D vector with custom z and w components
    inline Vec4f ToVec4f(float z, float w) const;

    // Extends to a 4D homogeneous direction vector (w = 0.0f, immune to translation)
    inline Vec4f ToVec4f_Direction(float z = 0.0f) const;
    
    // Extends to a 4D homogeneous point in space (w = 1.0f, affected by translation)
    inline Vec4f ToVec4f_Point(float z = 0.0f) const;
#pragma endregion
};