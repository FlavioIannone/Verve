#pragma once
#include <cmath>
#include <cassert>
#include "rasterizer/primitives/Mat4x4f.hpp"
#include "rasterizer/primitives/Vec3f.hpp"
#include "rasterizer/math_core/Align.hpp"
#include <rasterizer/simd/SimdMath.hpp>

// Represents a 16-byte aligned 4D Quaternion [x, y, z, w] for 3D rotations
union alignas(ALIGN_QUATERNION) Quaternion {
    struct {
        union {
            struct { float x, y, z; };
            Vec3f v; // Imaginary vector part [x, y, z] sharing bytes 0..11
        };
        float w;     // Real scalar part at byte offset 12
    };
    float arrView[4];
    __m128 simdView;

//#region Constructors

    // Initializes to the multiplicative identity quaternion [0, 0, 0, 1]
    inline Quaternion() {
        this->simdView = _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);
    }

    // Initializes components in little-endian lane order: [x, y, z, w]
    inline Quaternion(const float x, const float y, const float z, const float w) {
        this->simdView = _mm_setr_ps(x, y, z, w);
    }

    // Initializes from scalar real part w and 3D imaginary vector part v
    inline Quaternion(const float w, const Vec3f& v) {
        this->simdView = _mm_setr_ps(v.x, v.y, v.z, w);
    }

    // Initializes directly from an existing 128-bit SIMD register
    inline Quaternion(__m128 simd) {
        this->simdView = simd;
    }

    // Initializes components from a 4-element float array in order
    explicit inline Quaternion(float values[4]) {
        this->simdView = _mm_setr_ps(values[0], values[1], values[2], values[3]);
    }

    // Loads 4 contiguous floats from a 16-byte aligned memory address
    explicit inline Quaternion(const float *ptr) {
        this->simdView = _mm_load_ps(ptr);
    }

    // Default copy constructor
    inline Quaternion(const Quaternion& other) = default;

//#endregion

//#region AssignOperator

    // Default copy assignment operator
    inline Quaternion& operator=(const Quaternion& other) = default;

//#endregion

//#region Operators

    // Hamilton product (combines two 3D rotations: this * other)
    inline Quaternion MATH_VECTORCALL operator*(const Quaternion& other) const {
        Quaternion res = *this;
        return res *= other;
    }

    // Scales all quaternion components by a scalar
    inline Quaternion MATH_VECTORCALL operator*(const float scalar) const {
        Quaternion res = *this;
        return res *= scalar;
    }

    // In-place non-commutative Hamilton product using in-register SIMD shuffles and FMA
    inline Quaternion& MATH_VECTORCALL operator*=(const Quaternion& other) {
        // Permutes and sign-flips 'other' lanes for Hamilton cross-terms:
        // v1 = [w2, -z2, y2, -x2]
        __m128 v1 = _mm_xor_ps(
            _mm_shuffle_ps(other.simdView, other.simdView, _MM_SHUFFLE(0, 1, 2, 3)),
            _mm_setr_ps(0.0f, -0.0f, 0.0f, -0.0f));

        // v2 = [z2, w2, -x2, -y2]
        __m128 v2 = _mm_xor_ps(
            _mm_shuffle_ps(other.simdView, other.simdView, _MM_SHUFFLE(1, 0, 3, 2)),
            _mm_setr_ps(0.0f, 0.0f, -0.0f, -0.0f));

        // v3 = [-y2, x2, w2, -z2]
        __m128 v3 = _mm_xor_ps(
            _mm_shuffle_ps(other.simdView, other.simdView, _MM_SHUFFLE(2, 3, 0, 1)),
            _mm_setr_ps(-0.0f, 0.0f, 0.0f, -0.0f));

        // Broadcasts each component of 'this' across all 4 lanes directly in-register
        __m128 x1 = _mm_shuffle_ps(this->simdView, this->simdView, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 y1 = _mm_shuffle_ps(this->simdView, this->simdView, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 z1 = _mm_shuffle_ps(this->simdView, this->simdView, _MM_SHUFFLE(2, 2, 2, 2));
        __m128 w1 = _mm_shuffle_ps(this->simdView, this->simdView, _MM_SHUFFLE(3, 3, 3, 3));

        // Accumulates w1*q2 + x1*v1 + y1*v2 + z1*v3 via chained single-cycle FMA
        this->simdView = _mm_fmadd_ps(w1, other.simdView,
            _mm_fmadd_ps(x1, v1,
                _mm_fmadd_ps(y1, v2,
                    _mm_mul_ps(z1, v3)
                )
            )
        );

        return *this;
    }

    // In-place multiplication by a scalar
    inline Quaternion& MATH_VECTORCALL operator*=(const float scalar) {
        this->simdView = _mm_mul_ps(this->simdView, _mm_set1_ps(scalar));
        return *this;
    }

    // Rotates a 3D vector by this unit quaternion using Giesen's Euler-Rodrigues formula:
    // p' = p + w*t + (v x t), where t = 2 * (v x p)
    inline Vec3f MATH_VECTORCALL operator*(const Vec3f& other) const {
        __m128 b = _mm_setr_ps(other.x, other.y, other.z, 0.0f);

        // Pre-shuffles quaternion vector part v = [x, y, z] (reused in both cross products)
        __m128 a_yzx = _mm_shuffle_ps(this->simdView, this->simdView, _MM_SHUFFLE(3, 0, 2, 1));
        __m128 a_zxy = _mm_shuffle_ps(this->simdView, this->simdView, _MM_SHUFFLE(3, 1, 0, 2));

        // First cross product: v x b
        __m128 b_zxy = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 1, 0, 2));
        __m128 b_yzx = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1));
        __m128 term2 = _mm_mul_ps(a_zxy, b_yzx);
        __m128 t     = _mm_fmsub_ps(a_yzx, b_zxy, term2);

        // Doubles the first cross product: t = 2 * (v x b)
        t = _mm_add_ps(t, t);

        // Second cross product: cross = v x t
        __m128 t_zxy = _mm_shuffle_ps(t, t, _MM_SHUFFLE(3, 1, 0, 2));
        __m128 t_yzx = _mm_shuffle_ps(t, t, _MM_SHUFFLE(3, 0, 2, 1));
        term2        = _mm_mul_ps(a_zxy, t_yzx);
        __m128 cross = _mm_fmsub_ps(a_yzx, t_zxy, term2);

        // Broadcasts w in-register and computes res = t * w + (cross + b)
        __m128 w_bcast = _mm_shuffle_ps(this->simdView, this->simdView, _MM_SHUFFLE(3, 3, 3, 3));
        __m128 res     = _mm_fmadd_ps(t, w_bcast, _mm_add_ps(cross, b));

        alignas(16) float temp[4];
        _mm_store_ps(temp, res);

        return {temp[0], temp[1], temp[2]};
    }

    // Component-wise quaternion addition
    inline Quaternion MATH_VECTORCALL operator+(const Quaternion& other) const {
        Quaternion res = *this;
        return res += other;
    }

    // In-place component-wise quaternion addition
    inline Quaternion& MATH_VECTORCALL operator+=(const Quaternion& other) {
        this->simdView = _mm_add_ps(this->simdView, other.simdView);
        return *this;
    }

    // Component-wise quaternion subtraction
    inline Quaternion MATH_VECTORCALL operator-(const Quaternion& other) const {
        Quaternion res = *this;
        return res -= other;
    }

    // Unary negation (flips the sign bit of all 4 components; represents the same 3D rotation)
    inline Quaternion MATH_VECTORCALL operator-() const {
        return _mm_xor_ps(this->simdView, _mm_set1_ps(-0.0f));
    }

    // In-place component-wise quaternion subtraction
    inline Quaternion& MATH_VECTORCALL operator-=(const Quaternion& other) {
        this->simdView = _mm_sub_ps(this->simdView, other.simdView);
        return *this;
    }

    // Divides all components by a scalar
    inline Quaternion MATH_VECTORCALL operator/(const float scalar) const {
        Quaternion res = *this;
        return res /= scalar;
    }

    // In-place division by a scalar using full 23-bit IEEE-754 reciprocal
    inline Quaternion& MATH_VECTORCALL operator/=(const float scalar) {
        this->simdView = _mm_mul_ps(this->simdView, _mm_set1_ps(1.0f / scalar));
        return *this;
    }

//#endregion

//#region LogicalOperators

    // Exact component-wise equality check
    inline bool MATH_VECTORCALL operator==(const Quaternion& other) const {
        // Extracts the 4 comparison MSBs into a 4-bit integer mask (0x0F means all 4 lanes match)
        return _mm_movemask_ps(_mm_cmpeq_ps(this->simdView, other.simdView)) == 0x0F;
    }

    // Exact component-wise inequality check
    inline bool MATH_VECTORCALL operator!=(const Quaternion& other) const {
        return _mm_movemask_ps(_mm_cmpeq_ps(this->simdView, other.simdView)) != 0x0F;
    }

    // Checks if all 4 components match within a given epsilon tolerance (algebraic equality)
    inline bool MATH_VECTORCALL IsNearlyEqual(const Quaternion& other, float epsilon = 1e-4f) const {
        // Clears sign bit via ANDNOT with -0.0f to compute absolute component-wise difference
        __m128 absDiff = _mm_andnot_ps(_mm_set1_ps(-0.0f), _mm_sub_ps(this->simdView, other.simdView));
        return _mm_movemask_ps(_mm_cmple_ps(absDiff, _mm_set1_ps(epsilon))) == 0x0F;
    }

    // Checks if two unit quaternions represent the same 3D orientation in SO(3) (accounts for q == -q double cover)
    inline bool MATH_VECTORCALL IsSameRotation(const Quaternion& other, float epsilon = 1e-4f) const {
        __m128 dot    = _mm_dp_ps(this->simdView, other.simdView, 0xFF);
        __m128 absDot = _mm_andnot_ps(_mm_set1_ps(-0.0f), dot);
        return _mm_cvtss_f32(absDot) >= (1.0f - epsilon);
    }

//#endregion

//#region QuaternionOperations

    // Computes 4D dot product (x1*x2 + y1*y2 + z1*z2 + w1*w2)
    inline float MATH_VECTORCALL Dot(const Quaternion& other) const {
        // Mask 0xFF: multiplies all 4 lanes and broadcasts the sum to all lanes
        return _mm_cvtss_f32(_mm_dp_ps(this->simdView, other.simdView, 0xFF));
    }

    // Computes squared norm (x^2 + y^2 + z^2 + w^2)
    inline float MATH_VECTORCALL SquaredMagnitude() const {
        return this->Dot(*this);
    }

    // Computes Euclidean norm sqrt(x^2 + y^2 + z^2 + w^2)
    inline float MATH_VECTORCALL Magnitude() const {
        return std::sqrt(this->SquaredMagnitude());
    }

    // In-place fast normalization using hardware rsqrt + 1 Newton-Raphson step (assumes non-zero length)
    inline Quaternion& MATH_VECTORCALL NormalizeUnchecked() {
        assert(this->SquaredMagnitude() > 1e-8f);
        // Computes squared length x = Dot(q, q) broadcasted across all lanes
        __m128 x  = _mm_dp_ps(this->simdView, this->simdView, 0xFF);
        // Fast hardware approximation of 1 / sqrt(x) (~12-bit precision)
        __m128 y0 = _mm_rsqrt_ps(x);

        // One Newton-Raphson iteration restores full ~23-bit float precision:
        // y1 = -0.5 * y0 * (x * y0^2 - 3.0)
        const __m128 minus_half  = _mm_set1_ps(-0.5f);
        const __m128 minus_three = _mm_set1_ps(-3.0f);

        __m128 y0_sq         = _mm_mul_ps(y0, y0);
        __m128 minus_half_y0 = _mm_mul_ps(minus_half, y0);
        __m128 sub           = _mm_fmadd_ps(x, y0_sq, minus_three);

        __m128 invLen  = _mm_mul_ps(minus_half_y0, sub);
        this->simdView = _mm_mul_ps(this->simdView, invLen);

        return *this;
    }

    // Returns a normalized copy of this quaternion (assumes non-zero length)
    inline Quaternion MATH_VECTORCALL NormalizedUnchecked() const {
        Quaternion res = *this;
        res.NormalizeUnchecked();
        return res;
    }

    // In-place branchless safe normalization; falls back to Identity if squared magnitude <= 1e-8f
    inline Quaternion& MATH_VECTORCALL NormalizeSafe() {
        __m128 x  = _mm_dp_ps(this->simdView, this->simdView, 0xFF);
        __m128 y0 = _mm_rsqrt_ps(x);

        const __m128 minus_half  = _mm_set1_ps(-0.5f);
        const __m128 minus_three = _mm_set1_ps(-3.0f);

        __m128 y0_sq         = _mm_mul_ps(y0, y0);
        __m128 minus_half_y0 = _mm_mul_ps(minus_half, y0);
        __m128 sub           = _mm_fmadd_ps(x, y0_sq, minus_three);

        __m128 identity  = _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);
        __m128 invLen    = _mm_mul_ps(minus_half_y0, sub);
        this->simdView   = _mm_mul_ps(this->simdView, invLen);

        // Selects normalized result if x > 1e-8f, otherwise falls back to Identity
        __m128 validMask = _mm_cmpgt_ps(x, _mm_set1_ps(1e-8f));
        this->simdView   = _mm_blendv_ps(identity, this->simdView, validMask);

        return *this;
    }

    // Returns a safely normalized copy of this quaternion
    inline Quaternion MATH_VECTORCALL NormalizedSafe() const {
        Quaternion res = *this;
        res.NormalizeSafe();
        return res;
    }

    // In-place conjugate q* = [-x, -y, -z, w] (equals the inverse for unit quaternions)
    inline Quaternion& MATH_VECTORCALL Conjugate() {
        __m128 mask    = _mm_setr_ps(-0.0f, -0.0f, -0.0f, 0.0f);
        this->simdView = _mm_xor_ps(this->simdView, mask);
        return *this;
    }

    // Returns a conjugated copy of this quaternion
    inline Quaternion MATH_VECTORCALL Conjugated() const {
        Quaternion res = *this;
        res.Conjugate();
        return res;
    }

    // In-place multiplicative inverse q^-1 = q* / ||q||^2 (assumes non-zero norm)
    inline Quaternion& MATH_VECTORCALL InverseUnchecked() {
        assert(this->SquaredMagnitude() > 1e-8f);
        __m128 normSq  = _mm_dp_ps(this->simdView, this->simdView, 0xFF);
        __m128 conj    = _mm_xor_ps(this->simdView, _mm_setr_ps(-0.0f, -0.0f, -0.0f, 0.0f));
        this->simdView = _mm_div_ps(conj, normSq);
        return *this;
    }

    // Returns the multiplicative inverse q^-1 = q* / ||q||^2 (assumes non-zero norm)
    inline Quaternion MATH_VECTORCALL InversedUnchecked() const {
        Quaternion res = *this;
        res.InverseUnchecked();
        return res;
    }

    // In-place branchless multiplicative inverse; falls back to Identity if ||q||^2 <= 1e-8f
    inline Quaternion& MATH_VECTORCALL InverseSafe() {
        __m128 normSq    = _mm_dp_ps(this->simdView, this->simdView, 0xFF);
        __m128 conj      = _mm_xor_ps(this->simdView, _mm_setr_ps(-0.0f, -0.0f, -0.0f, 0.0f));
        __m128 inv       = _mm_div_ps(conj, normSq);
        __m128 identity  = _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);
        __m128 validMask = _mm_cmpgt_ps(normSq, _mm_set1_ps(1e-8f));
        this->simdView   = _mm_blendv_ps(identity, inv, validMask);
        return *this;
    }

    // Returns a branchless safe multiplicative inverse of this quaternion
    inline Quaternion MATH_VECTORCALL InversedSafe() const {
        Quaternion res = *this;
        res.InverseSafe();
        return res;
    }

    // Extracts the rotation axis and angle (in radians, [0, pi]) from this unit quaternion
    inline void ToAxisAngle(Vec3f& outAxis, float& outAngleRadians) const {
        // Clamps w to [-1, 1] to prevent std::acos domain NaNs from floating-point drift
        const float clampedW  = (this->w < -1.0f) ? -1.0f : ((this->w > 1.0f) ? 1.0f : this->w);
        outAngleRadians       = 2.0f * std::acos(clampedW);
        const float sinHalfSq = 1.0f - clampedW * clampedW;

        if (sinHalfSq > 1e-6f) {
            const float invSinHalf = 1.0f / std::sqrt(sinHalfSq);
            outAxis = this->v * invSinHalf;
        } else {
            // Near zero rotation angle: any unit axis is valid
            outAxis = Vec3f(1.0f, 0.0f, 0.0f);
        }
    }

//#endregion

//#region FactoryMethods

    // Returns the multiplicative identity quaternion [0, 0, 0, 1]
    static inline Quaternion Identity() {
        return Quaternion();
    }

    // Creates a rotation quaternion from a normalized 3D axis and an angle in radians
    static inline Quaternion FromAxisAngle(const Vec3f& axis, const float angle_radians) {
        const float halfAngle = angle_radians * 0.5f;
        const float sin       = std::sin(halfAngle);
        const float cos       = std::cos(halfAngle);

        return {cos, axis * sin};
    }

    // Creates a counter-clockwise rotation quaternion around the X-axis (pitch, in radians)
    static inline Quaternion FromAngleX(const float angle_radians) {
        const float halfAngle = angle_radians * 0.5f;
        const float sin       = std::sin(halfAngle);
        const float cos       = std::cos(halfAngle);
        return {sin, 0.0f, 0.0f, cos};
    }

    // Creates a counter-clockwise rotation quaternion around the Y-axis (yaw, in radians)
    static inline Quaternion FromAngleY(const float angle_radians) {
        const float halfAngle = angle_radians * 0.5f;
        const float sin       = std::sin(halfAngle);
        const float cos       = std::cos(halfAngle);
        return {0.0f, sin, 0.0f, cos};
    }

    // Creates a counter-clockwise rotation quaternion around the Z-axis (roll, in radians)
    static inline Quaternion FromAngleZ(const float angle_radians) {
        const float halfAngle = angle_radians * 0.5f;
        const float sin       = std::sin(halfAngle);
        const float cos       = std::cos(halfAngle);
        return {0.0f, 0.0f, sin, cos};
    }

    // Builds a quaternion from Euler angles (pitch=x, yaw=y, roll=z in radians) matching Rz * Ry * Rx order
    static inline Quaternion MATH_VECTORCALL FromEulerAngles(const Vec3f& angles) {
        const float hx = angles.x * 0.5f;
        const float hy = angles.y * 0.5f;
        const float hz = angles.z * 0.5f;

        // Evaluates sin/cos pairs on the same angle to enable compiler sincosf hardware fusion
        const float sx = std::sin(hx), cx = std::cos(hx);
        const float sy = std::sin(hy), cy = std::cos(hy);
        const float sz = std::sin(hz), cz = std::cos(hz);

        // Packs half-angle combinations into SIMD lanes:
        // term1 = [sx*cy*cz, cx*sy*cz, cx*cy*sz, cx*cy*cz]
        // term2 = [cx*sy*sz, sx*cy*sz, sx*sy*cz, sx*sy*sz]
        __m128 term1 = _mm_mul_ps(_mm_mul_ps(_mm_setr_ps(sx, cx, cx, cx), _mm_setr_ps(cy, sy, cy, cy)), _mm_setr_ps(cz, cz, sz, cz));
        __m128 term2 = _mm_mul_ps(_mm_mul_ps(_mm_setr_ps(cx, sx, sx, sx), _mm_setr_ps(sy, cy, sy, sy)), _mm_setr_ps(sz, sz, cz, sz));

        // Applies alternating signs [-, +, -, +] to term2 and sums with term1
        __m128 signMask = _mm_setr_ps(-0.0f, 0.0f, -0.0f, 0.0f);
        return _mm_add_ps(term1, _mm_xor_ps(term2, signMask));
    }

    // Computes shortest-arc rotation from unit vector 'from' to unit vector 'to' (Stan Melax's method)
    static inline Quaternion MATH_VECTORCALL FromToRotation(const Vec3f& from, const Vec3f& to) {
        const float d = from.Dot(to);

        // Vectors are nearly parallel: return Identity
        if (d >= 0.999999f) {
            return Identity();
        }

        // Vectors are nearly antipodal (180 degrees apart): pick an orthogonal rotation axis
        if (d < -0.999999f) {
            Vec3f axis = (std::abs(from.x) < 0.9f)
                ? Vec3f(1.0f, 0.0f, 0.0f).Cross(from)
                : Vec3f(0.0f, 1.0f, 0.0f).Cross(from);
            axis.Normalize();
            return {axis.x, axis.y, axis.z, 0.0f};
        }

        // General case: q = [from x to, 1 + dot(from, to)] normalized via fast hardware rsqrt
        const Vec3f c = from.Cross(to);
        return Quaternion(c.x, c.y, c.z, 1.0f + d).NormalizeUnchecked();
    }

    // Extracts a unit quaternion from a column-major 4x4 rotation matrix using Shepperd's method (1978)
    static inline Quaternion MATH_VECTORCALL FromMat4x4f(const Mat4x4f& m) {
        const float m00 = m.c0.x, m10 = m.c0.y, m20 = m.c0.z;
        const float m01 = m.c1.x, m11 = m.c1.y, m21 = m.c1.z;
        const float m02 = m.c2.x, m12 = m.c2.y, m22 = m.c2.z;

        const float trace = m00 + m11 + m22;

        // Selects the largest denominator component among w, x, y, z to guarantee numerical stability
        if (trace > 0.0f) {
            const float s    = std::sqrt(trace + 1.0f) * 2.0f; // s = 4 * w
            const float invS = 1.0f / s;
            return {
                (m21 - m12) * invS,
                (m02 - m20) * invS,
                (m10 - m01) * invS,
                0.25f * s
            };
        } else if ((m00 > m11) && (m00 > m22)) {
            const float s    = std::sqrt(1.0f + m00 - m11 - m22) * 2.0f; // s = 4 * x
            const float invS = 1.0f / s;
            return {
                0.25f * s,
                (m10 + m01) * invS,
                (m02 + m20) * invS,
                (m21 - m12) * invS
            };
        } else if (m11 > m22) {
            const float s    = std::sqrt(1.0f + m11 - m00 - m22) * 2.0f; // s = 4 * y
            const float invS = 1.0f / s;
            return {
                (m10 + m01) * invS,
                0.25f * s,
                (m21 + m12) * invS,
                (m02 - m20) * invS
            };
        } else {
            const float s    = std::sqrt(1.0f + m22 - m00 - m11) * 2.0f; // s = 4 * z
            const float invS = 1.0f / s;
            return {
                (m02 + m20) * invS,
                (m21 + m12) * invS,
                0.25f * s,
                (m10 - m01) * invS
            };
        }
    }

    // Builds a right-handed orientation quaternion looking along 'forward' with the given 'up' reference
    static inline Quaternion MATH_VECTORCALL LookRotation(const Vec3f& forward, const Vec3f& up) {
        // Right-handed camera convention: local -Z is forward, so +Z basis axis points along -forward
        const Vec3f f_axis = (-forward).Normalized();
        const Vec3f r_axis = up.Cross(f_axis).Normalized();
        const Vec3f u_axis = f_axis.Cross(r_axis);

        Mat4x4f rotMat;
        rotMat.c0 = Vec4f(r_axis.x, r_axis.y, r_axis.z, 0.0f);
        rotMat.c1 = Vec4f(u_axis.x, u_axis.y, u_axis.z, 0.0f);
        rotMat.c2 = Vec4f(f_axis.x, f_axis.y, f_axis.z, 0.0f);
        rotMat.c3 = Vec4f(0.0f, 0.0f, 0.0f, 1.0f);

        return FromMat4x4f(rotMat);
    }

    // Computes the shortest angular distance in radians [0, pi] between two unit quaternions
    static inline float MATH_VECTORCALL Angle(const Quaternion& a, const Quaternion& b) {
        __m128 dot       = _mm_dp_ps(a.simdView, b.simdView, 0xFF);
        __m128 absDot    = _mm_andnot_ps(_mm_set1_ps(-0.0f), dot);
        // Clamps |a . b| to 1.0f in hardware before acos to guard against floating-point overshoot
        float clampedDot = _mm_cvtss_f32(_mm_min_ss(absDot, _mm_set_ss(1.0f)));
        return 2.0f * std::acos(clampedDot);
    }

    // Normalized Linear Interpolation (Nlerp) with branchless shortest-path sign flip
    static inline Quaternion MATH_VECTORCALL Nlerp(const Quaternion& a, const Quaternion& b, float t) {
        __m128 dot  = _mm_dp_ps(a.simdView, b.simdView, 0xFF);
        __m128 mask = _mm_and_ps(dot, _mm_set1_ps(-0.0f));
        Quaternion b1 = _mm_xor_ps(b.simdView, mask);

        Quaternion n = _mm_fmadd_ps((b1 - a).simdView, _mm_set1_ps(t), a.simdView);
        return n.NormalizeSafe();
    }

    // Spherical Linear Interpolation (Slerp) with shortest-path hemisphere selection and Nlerp fallback
    static inline Quaternion MATH_VECTORCALL Slerp(const Quaternion& a, const Quaternion& b, float t) {
        __m128 dot        = _mm_dp_ps(a.simdView, b.simdView, 0xFF);
        __m128 absDot     = _mm_andnot_ps(_mm_set1_ps(-0.0f), dot);
        float absCosTheta = _mm_cvtss_f32(absDot);

        // Falls back to Nlerp when angle is near zero to avoid 1 / sin(theta) division by zero
        if (absCosTheta > 0.9995f) {
            return Nlerp(a, b, t);
        }

        // Flips b into b1 if dot < 0 to interpolate along the shortest geodesic arc on S^3
        __m128 mask   = _mm_and_ps(dot, _mm_set1_ps(-0.0f));
        Quaternion b1 = _mm_xor_ps(b.simdView, mask);

        float theta       = std::acos(absCosTheta);
        float invSinTheta = 1.0f / std::sqrt(1.0f - absCosTheta * absCosTheta);
        float w1          = std::sin((1.0f - t) * theta) * invSinTheta;
        float w2          = std::sin(t * theta) * invSinTheta;

        return _mm_fmadd_ps(_mm_set1_ps(w1), a.simdView, _mm_mul_ps(_mm_set1_ps(w2), b1.simdView));
    }

//#endregion

//#region ConversionOperators

    // Converts this unit quaternion into a column-major 4x4 rotation matrix purely in SIMD registers
    inline Mat4x4f MATH_VECTORCALL ToMat4x4f() const {
        Mat4x4f res;

        const __m128 q = this->simdView;
        // Computes [2x, 2y, 2z, 2w] and zeroes out lane 3 -> q2 = [x2, y2, z2, 0.0f]
        __m128 q2 = _mm_add_ps(q, q);
        q2        = _mm_blend_ps(q2, _mm_setzero_ps(), 0x8);

        // Column 0: [1 - yy2 - zz2, xy2 + wz2, xz2 - wy2, 0]
        // c0 = [1, 0, 0, 0] + [y, x, x, w]*[-y2, y2, z2, 0] + [z, w, w, w]*[-z2, z2, -y2, 0]
        __m128 c0_a = _mm_shuffle_ps(q, q, _MM_SHUFFLE(3, 0, 0, 1));
        __m128 c0_b = _mm_xor_ps(_mm_shuffle_ps(q2, q2, _MM_SHUFFLE(3, 2, 1, 1)), _mm_setr_ps(-0.0f, 0.0f, 0.0f, 0.0f));
        __m128 c0_c = _mm_shuffle_ps(q, q, _MM_SHUFFLE(3, 3, 3, 2));
        __m128 c0_d = _mm_xor_ps(_mm_shuffle_ps(q2, q2, _MM_SHUFFLE(3, 1, 2, 2)), _mm_setr_ps(-0.0f, 0.0f, -0.0f, 0.0f));
        res.c0      = _mm_fmadd_ps(c0_a, c0_b, _mm_fmadd_ps(c0_c, c0_d, _mm_setr_ps(1.0f, 0.0f, 0.0f, 0.0f)));

        // Column 1: [xy2 - wz2, 1 - xx2 - zz2, yz2 + wx2, 0]
        // c1 = [0, 1, 0, 0] + [x, x, y, w]*[y2, -x2, z2, 0] + [w, z, w, w]*[-z2, -z2, x2, 0]
        __m128 c1_a = _mm_shuffle_ps(q, q, _MM_SHUFFLE(3, 1, 0, 0));
        __m128 c1_b = _mm_xor_ps(_mm_shuffle_ps(q2, q2, _MM_SHUFFLE(3, 2, 0, 1)), _mm_setr_ps(0.0f, -0.0f, 0.0f, 0.0f));
        __m128 c1_c = _mm_shuffle_ps(q, q, _MM_SHUFFLE(3, 3, 2, 3));
        __m128 c1_d = _mm_xor_ps(_mm_shuffle_ps(q2, q2, _MM_SHUFFLE(3, 0, 2, 2)), _mm_setr_ps(-0.0f, -0.0f, 0.0f, 0.0f));
        res.c1      = _mm_fmadd_ps(c1_a, c1_b, _mm_fmadd_ps(c1_c, c1_d, _mm_setr_ps(0.0f, 1.0f, 0.0f, 0.0f)));

        // Column 2: [xz2 + wy2, yz2 - wx2, 1 - xx2 - yy2, 0]
        // c2 = [0, 0, 1, 0] + [x, y, x, w]*[z2, z2, -x2, 0] + [w, w, y, w]*[y2, -x2, -y2, 0]
        __m128 c2_a = _mm_shuffle_ps(q, q, _MM_SHUFFLE(3, 0, 1, 0));
        __m128 c2_b = _mm_xor_ps(_mm_shuffle_ps(q2, q2, _MM_SHUFFLE(3, 0, 2, 2)), _mm_setr_ps(0.0f, 0.0f, -0.0f, 0.0f));
        __m128 c2_c = _mm_shuffle_ps(q, q, _MM_SHUFFLE(3, 1, 3, 3));
        __m128 c2_d = _mm_xor_ps(_mm_shuffle_ps(q2, q2, _MM_SHUFFLE(3, 1, 0, 1)), _mm_setr_ps(0.0f, -0.0f, -0.0f, 0.0f));
        res.c2      = _mm_fmadd_ps(c2_a, c2_b, _mm_fmadd_ps(c2_c, c2_d, _mm_setr_ps(0.0f, 0.0f, 1.0f, 0.0f)));

        // Column 3: Homogeneous identity column [0, 0, 0, 1]
        res.c3      = _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);

        return res;
    }

//#endregion

//#region IndexingOperators

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
};