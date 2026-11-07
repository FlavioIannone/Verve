#pragma once
#include <cmath>
#include "rasterizer/primitives/VecConfig.hpp"
#include "rasterizer/primitives/VecUtils.hpp"
#include "rasterizer/primitives/Vec4f.hpp"
#include "rasterizer/primitives/Vec3f.hpp"
#include "rasterizer/math_core/Align.hpp"
#include <rasterizer/simd/SimdMath.hpp>

// Internal container holding 12 sub-determinants (2x2 minors) packed into 3 SIMD registers
struct Minors2x2 {
    Vec4f s_0123;     // Minors from cols 0 & 1: [s0, s1, s2, s3]
    Vec4f s_45_t_01;  // Mixed minors from cols (0,1) & (2,3): [s4, s5, t0, t1]
    Vec4f t_2345;     // Minors from cols 2 & 3: [t2, t3, t4, t5]
};

// Internal container for the 6 shuffled minor vectors used in Laplace cofactor expansion
struct MinorCombinationVectors {
    Vec4f t1, t2, t3; // Shuffled 2x2 minors for computing Cofactor Columns 0 and 1
    Vec4f s1, s2, s3; // Shuffled 2x2 minors for computing Cofactor Columns 2 and 3
};

// Represents a 4x4 Column-Major Matrix aligned for SIMD operations
union alignas(ALIGN_MAT4X4F) Mat4x4f
{
    Vec4f cols[4];
    struct { Vec4f c0, c1, c2, c3; };
    float mat[16];

//#region Constructors
    // Default constructor: initializes to the 4x4 Identity Matrix
    inline Mat4x4f() {
        c0 = Vec4f(1.0f, 0.0f, 0.0f, 0.0f);
        c1 = Vec4f(0.0f, 1.0f, 0.0f, 0.0f);
        c2 = Vec4f(0.0f, 0.0f, 1.0f, 0.0f);
        c3 = Vec4f(0.0f, 0.0f, 0.0f, 1.0f);
    }

    // Default copy constructor
    inline Mat4x4f(const Mat4x4f& other) = default;

    // Initializes a diagonal matrix with the given scalar along the main diagonal
    inline Mat4x4f(float diagonal) {
        c0 = Vec4f(diagonal, 0.0f, 0.0f, 0.0f);
        c1 = Vec4f(0.0f, diagonal, 0.0f, 0.0f);
        c2 = Vec4f(0.0f, 0.0f, diagonal, 0.0f);
        c3 = Vec4f(0.0f, 0.0f, 0.0f, diagonal);
    }

    // Initializes matrix from 4 column vectors
    inline Mat4x4f(const Vec4f& c0, const Vec4f& c1, const Vec4f& c2, const Vec4f& c3) {
        this->c0 = c0;
        this->c1 = c1;
        this->c2 = c2;
        this->c3 = c3;
    }

    // Loads 16 contiguous floats in column-major order from a raw pointer
    explicit inline Mat4x4f(const float* ptr) {
        c0 = Vec4f(ptr[0], ptr[1], ptr[2], ptr[3]);
        c1 = Vec4f(ptr[4], ptr[5], ptr[6], ptr[7]);
        c2 = Vec4f(ptr[8], ptr[9], ptr[10], ptr[11]);
        c3 = Vec4f(ptr[12], ptr[13], ptr[14], ptr[15]);
    }
//#endregion

//#region AssignOperator
    // Default copy assignment operator
    inline Mat4x4f &operator=(const Mat4x4f& other) = default;
//#endregion

//#region MatrixOperators
    // Multiplies Matrix by a 4D column vector (M * v) using linear combination of columns
    inline Vec4f MATH_VECTORCALL operator*(const Vec4f &other) const {
        // Broadcasts each vector component and accumulates scaled columns via FMA:
        // res = w*c3 + z*c2 + y*c1 + x*c0
        Vec4f res(_mm_mul_ps(_mm_set1_ps(other.w), this->c3.simdView));

        res = res.fmadd(_mm_set1_ps(other.z), this->c2.simdView);
        res = res.fmadd(_mm_set1_ps(other.y), this->c1.simdView);
        res = res.fmadd(_mm_set1_ps(other.x), this->c0.simdView);
        
        return res;
    }

    // Transforms a 3D Point assuming homogeneous w = 1.0f (applies translation column c3)
    inline Vec4f MATH_VECTORCALL TransformPoint(const Vec3f& other) const {
        // Starts directly from c3 since w = 1.0f makes the initial multiplication redundant
        Vec4f res = this->c3;
        
        res = res.fmadd(_mm_set1_ps(other.z), this->c2.simdView);
        res = res.fmadd(_mm_set1_ps(other.y), this->c1.simdView);
        res = res.fmadd(_mm_set1_ps(other.x), this->c0.simdView);
        
        return res;
    }
    
    // Transforms a 3D Direction vector assuming homogeneous w = 0.0f (ignores translation column c3)
    inline Vec4f MATH_VECTORCALL TransformDirection(const Vec3f& other) const {
        Vec4f res = Vec4f();
        
        res = res.fmadd(_mm_set1_ps(other.z), this->c2.simdView);
        res = res.fmadd(_mm_set1_ps(other.y), this->c1.simdView);
        res = res.fmadd(_mm_set1_ps(other.x), this->c0.simdView);
        
        return res;
    }

    // Matrix-Matrix multiplication (transforms each column of 'other' by this matrix)
    inline Mat4x4f MATH_VECTORCALL operator*(const Mat4x4f& other) const {
        Mat4x4f res;
        res.c0 = (*this) * other.c0;
        res.c1 = (*this) * other.c1;
        res.c2 = (*this) * other.c2;
        res.c3 = (*this) * other.c3;
        return res;
    }
    
    // In-place Matrix-Matrix multiplication
    inline Mat4x4f& MATH_VECTORCALL operator*=(const Mat4x4f& other) {
        *this = *this * other;
        return *this;
    }

    // Returns a transposed copy of this matrix (swaps rows and columns)
    inline Mat4x4f MATH_VECTORCALL Transposed() const {
        Mat4x4f res = *this;
        // Hardware 4x4 in-register SIMD transpose macro (interleaves low/high pairs)
        _MM_TRANSPOSE4_PS(res.c0.simdView, res.c1.simdView, res.c2.simdView, res.c3.simdView);
        return res;
    }

    // In-place matrix transposition
    inline Mat4x4f& MATH_VECTORCALL Transpose() {
        _MM_TRANSPOSE4_PS(this->c0.simdView, this->c1.simdView, this->c2.simdView, this->c3.simdView);
        return *this;
    }

    // Computes the determinant of a general 4x4 matrix via Laplace expansion along Column 0
    inline float MATH_VECTORCALL DeterminantGeneral() const {
        const Minors2x2 minors = this->_ComputeMinors2x2();

        // Shuffles only the t-minors required to build Column 0 cofactors
        const Vec4f t1 = _mm_shuffle_ps(minors.t_2345.simdView, minors.t_2345.simdView, _MM_SHUFFLE(1, 2, 3, 3));
        const simde__m128 temp_t2 = _mm_shuffle_ps(minors.t_2345.simdView, minors.s_45_t_01.simdView, _MM_SHUFFLE(3, 0, 0, 0));
        const Vec4f t2 = _mm_shuffle_ps(minors.t_2345.simdView, temp_t2, _MM_SHUFFLE(3, 0, 0, 2));
        const simde__m128 temp_t3 = _mm_shuffle_ps(minors.t_2345.simdView, minors.s_45_t_01.simdView, _MM_SHUFFLE(3, 2, 1, 1));
        const Vec4f t3 = _mm_shuffle_ps(temp_t3, minors.s_45_t_01.simdView, _MM_SHUFFLE(2, 2, 3, 0));

        // Permutes Column 1 components for 3x3 cofactor expansion
        const Vec4f v1_c1 = _mm_shuffle_ps(this->c1.simdView, this->c1.simdView, _MM_SHUFFLE(0, 0, 0, 1));
        const Vec4f v2_c1 = _mm_shuffle_ps(this->c1.simdView, this->c1.simdView, _MM_SHUFFLE(1, 1, 2, 2));
        const Vec4f v3_c1 = _mm_shuffle_ps(this->c1.simdView, this->c1.simdView, _MM_SHUFFLE(2, 3, 3, 3));

        // Alternating cofactor sign mask: [+, -, +, -]
        const simde__m128 sign_mask0 = _mm_setr_ps(0.0f, -0.0f, 0.0f, -0.0f);

        // Assembles Column 0 of the Cofactor Matrix: [C_00, C_10, C_20, C_30]
        const Vec4f col0_cofactors = this->_AssembleAdjugateColumn(v1_c1, v2_c1, v3_c1, t1, t2, t3, sign_mask0);

        // det(M) = Dot(Column 0, Cofactor Column 0)
        return this->c0.Dot(col0_cofactors);
    }

    // Computes the inverse of a general 4x4 matrix; returns false if singular/non-invertible
    inline bool MATH_VECTORCALL InverseGeneral(Mat4x4f& out_inverse) const {
        // 1. Computes Adjugate Matrix adj(M) = C^T
        const Mat4x4f adj = this->_Adjugate();

        // 2. Extracts Row 0 of M [m_00, m_01, m_02, m_03] to dot with Adjugate Col 0 [C_00, C_01, C_02, C_03]
        const simde__m128 low_01 = _mm_unpacklo_ps(this->c0.simdView, this->c1.simdView); // [m_00, m_01, m_10, m_11]
        const simde__m128 low_23 = _mm_unpacklo_ps(this->c2.simdView, this->c3.simdView); // [m_02, m_03, m_12, m_13]
        const Vec4f row0 = _mm_movelh_ps(low_01, low_23);                                 // [m_00, m_01, m_02, m_03]

        const float det = row0.Dot(adj.c0);

        // 3. Rejects singular matrices or NaN determinants (negated condition catches NaNs)
        if (!(std::abs(det) >= 1e-8f)) {
            return false;
        }

        // 4. Scales Adjugate columns by 1 / det: M^-1 = adj(M) / det(M)
        const simde__m128 inv_det_simd = _mm_set1_ps(1.0f / det);
        out_inverse.c0 = _mm_mul_ps(adj.c0.simdView, inv_det_simd);
        out_inverse.c1 = _mm_mul_ps(adj.c1.simdView, inv_det_simd);
        out_inverse.c2 = _mm_mul_ps(adj.c2.simdView, inv_det_simd);
        out_inverse.c3 = _mm_mul_ps(adj.c3.simdView, inv_det_simd);

        return true;
    }

    // Returns the inverted matrix, or falls back to Identity if the matrix is singular
    inline Mat4x4f MATH_VECTORCALL Inversed() const {
        Mat4x4f inv;
        if (this->InverseGeneral(inv)) {
            return inv;
        }
        return Identity();
    }
//#endregion

//#region InverseMatrixHelpers
    // Computes all 12 2x2 sub-determinants (minors) packed into 3 SIMD registers via FMA
    inline Minors2x2 MATH_VECTORCALL _ComputeMinors2x2() const {
        Minors2x2 minors;

        // Computes 2x2 minors (a*d - b*c) for columns (c0, c1): [s0, s1, s2, s3]
        minors.s_0123 = _mm_fmsub_ps(
            _mm_shuffle_ps(this->c0.simdView, this->c0.simdView, _MM_SHUFFLE(1, 0, 0, 0)),
            _mm_shuffle_ps(this->c1.simdView, this->c1.simdView, _MM_SHUFFLE(2, 3, 2, 1)),
            _mm_mul_ps(
                _mm_shuffle_ps(this->c0.simdView, this->c0.simdView, _MM_SHUFFLE(2, 3, 2, 1)),
                _mm_shuffle_ps(this->c1.simdView, this->c1.simdView, _MM_SHUFFLE(1, 0, 0, 0))
            )
        );

        // Computes remaining 2 minors of (c0, c1) and first 2 minors of (c2, c3): [s4, s5, t0, t1]
        minors.s_45_t_01 = _mm_fmsub_ps(
            _mm_shuffle_ps(this->c0.simdView, this->c2.simdView, _MM_SHUFFLE(0, 0, 2, 1)),
            _mm_shuffle_ps(this->c1.simdView, this->c3.simdView, _MM_SHUFFLE(2, 1, 3, 3)),
            _mm_mul_ps(
                _mm_shuffle_ps(this->c0.simdView, this->c2.simdView, _MM_SHUFFLE(2, 1, 3, 3)),
                _mm_shuffle_ps(this->c1.simdView, this->c3.simdView, _MM_SHUFFLE(0, 0, 2, 1))
            )
        );

        // Computes remaining 4 minors for columns (c2, c3): [t2, t3, t4, t5]
        minors.t_2345 = _mm_fmsub_ps(
            _mm_shuffle_ps(this->c2.simdView, this->c2.simdView, _MM_SHUFFLE(2, 1, 1, 0)),
            _mm_shuffle_ps(this->c3.simdView, this->c3.simdView, _MM_SHUFFLE(3, 3, 2, 3)),
            _mm_mul_ps(
                _mm_shuffle_ps(this->c2.simdView, this->c2.simdView, _MM_SHUFFLE(3, 3, 2, 3)),
                _mm_shuffle_ps(this->c3.simdView, this->c3.simdView, _MM_SHUFFLE(2, 1, 1, 0))
            )
        );

        return minors;
    }

    // Rearranges packed 2x2 minors into the 6 operand vectors needed for 3x3 cofactor expansion
    inline MinorCombinationVectors MATH_VECTORCALL _PrepareMinorVectors(const Minors2x2& minors) const {
        MinorCombinationVectors combo;

        // Single-stage shuffle for t1: [t5, t5, t4, t3]
        combo.t1 = _mm_shuffle_ps(minors.t_2345.simdView, minors.t_2345.simdView, _MM_SHUFFLE(1, 2, 3, 3));

        // Two-stage shuffle across registers to assemble t2: [t4, t2, t2, t1]
        const simde__m128 temp_t2 = _mm_shuffle_ps(minors.t_2345.simdView, minors.s_45_t_01.simdView, _MM_SHUFFLE(3, 0, 0, 0));
        combo.t2 = _mm_shuffle_ps(minors.t_2345.simdView, temp_t2, _MM_SHUFFLE(3, 0, 0, 2));

        // Two-stage shuffle across registers to assemble t3: [t3, t1, t0, t0]
        const simde__m128 temp_t3 = _mm_shuffle_ps(minors.t_2345.simdView, minors.s_45_t_01.simdView, _MM_SHUFFLE(3, 2, 1, 1));
        combo.t3 = _mm_shuffle_ps(temp_t3, minors.s_45_t_01.simdView, _MM_SHUFFLE(2, 2, 3, 0));

        // Two-stage shuffle across registers to assemble s1: [s5, s5, s4, s3]
        const simde__m128 temp_s1 = _mm_shuffle_ps(minors.s_45_t_01.simdView, minors.s_0123.simdView, _MM_SHUFFLE(3, 0, 0, 0));
        combo.s1 = _mm_shuffle_ps(minors.s_45_t_01.simdView, temp_s1, _MM_SHUFFLE(3, 0, 1, 1));

        // Two-stage shuffle across registers to assemble s2: [s4, s2, s2, s1]
        const simde__m128 temp_s2 = _mm_shuffle_ps(minors.s_0123.simdView, minors.s_45_t_01.simdView, _MM_SHUFFLE(0, 0, 2, 2));
        combo.s2 = _mm_shuffle_ps(temp_s2, minors.s_0123.simdView, _MM_SHUFFLE(1, 2, 0, 2));

        // Single-stage shuffle for s3: [s3, s1, s0, s0]
        combo.s3 = _mm_shuffle_ps(minors.s_0123.simdView, minors.s_0123.simdView, _MM_SHUFFLE(0, 0, 1, 3));

        return combo;
    }

    // Evaluates 4 cofactors simultaneously using chained FMA and bitwise sign-flipping
    inline Vec4f MATH_VECTORCALL _AssembleAdjugateColumn(
        const Vec4f& v1_col,
        const Vec4f& v2_col,
        const Vec4f& v3_col,
        const Vec4f& M1,
        const Vec4f& M2,
        const Vec4f& M3,
        const simde__m128& sign_mask
    ) const {
        // Evaluates 3x3 determinant expansion: (v3 * M3) + ((v1 * M1) - (v2 * M2))
        const Vec4f raw_column = _mm_fmadd_ps(
            v3_col.simdView, M3.simdView,
            _mm_fmsub_ps(v1_col.simdView, M1.simdView, _mm_mul_ps(v2_col.simdView, M2.simdView))
        );

        // Applies checkerboard cofactor signs (+ - + - or - + - +) via bitwise XOR
        return _mm_xor_ps(raw_column.simdView, sign_mask);
    }

    // Computes the full Adjugate Matrix (transpose of the Cofactor Matrix, C^T)
    inline Mat4x4f MATH_VECTORCALL _Adjugate() const {
        // 1. Computes packed 2x2 sub-determinants
        const Minors2x2 minors = this->_ComputeMinors2x2();

        // 2. Prepares shuffled minor operand vectors
        const MinorCombinationVectors combo = this->_PrepareMinorVectors(minors);

        // 3. Shuffles matrix columns to align with 2x2 minor combinations
        const Vec4f v1_c1 = _mm_shuffle_ps(this->c1.simdView, this->c1.simdView, _MM_SHUFFLE(0, 0, 0, 1));
        const Vec4f v2_c1 = _mm_shuffle_ps(this->c1.simdView, this->c1.simdView, _MM_SHUFFLE(1, 1, 2, 2));
        const Vec4f v3_c1 = _mm_shuffle_ps(this->c1.simdView, this->c1.simdView, _MM_SHUFFLE(2, 3, 3, 3));

        const Vec4f v1_c0 = _mm_shuffle_ps(this->c0.simdView, this->c0.simdView, _MM_SHUFFLE(0, 0, 0, 1));
        const Vec4f v2_c0 = _mm_shuffle_ps(this->c0.simdView, this->c0.simdView, _MM_SHUFFLE(1, 1, 2, 2));
        const Vec4f v3_c0 = _mm_shuffle_ps(this->c0.simdView, this->c0.simdView, _MM_SHUFFLE(2, 3, 3, 3));

        const Vec4f v1_c3 = _mm_shuffle_ps(this->c3.simdView, this->c3.simdView, _MM_SHUFFLE(0, 0, 0, 1));
        const Vec4f v2_c3 = _mm_shuffle_ps(this->c3.simdView, this->c3.simdView, _MM_SHUFFLE(1, 1, 2, 2));
        const Vec4f v3_c3 = _mm_shuffle_ps(this->c3.simdView, this->c3.simdView, _MM_SHUFFLE(2, 3, 3, 3));

        const Vec4f v1_c2 = _mm_shuffle_ps(this->c2.simdView, this->c2.simdView, _MM_SHUFFLE(0, 0, 0, 1));
        const Vec4f v2_c2 = _mm_shuffle_ps(this->c2.simdView, this->c2.simdView, _MM_SHUFFLE(1, 1, 2, 2));
        const Vec4f v3_c2 = _mm_shuffle_ps(this->c2.simdView, this->c2.simdView, _MM_SHUFFLE(2, 3, 3, 3));

        // Checkerboard sign masks for cofactor columns
        const simde__m128 sign_mask0 = _mm_setr_ps(0.0f, -0.0f, 0.0f, -0.0f); // Pattern: [+, -, +, -]
        const simde__m128 sign_mask1 = _mm_setr_ps(-0.0f, 0.0f, -0.0f, 0.0f); // Pattern: [-, +, -, +]

        // 4. Computes the 4 columns of the Cofactor Matrix C (which correspond to rows of Adjugate C^T)
        simde__m128 r0 = this->_AssembleAdjugateColumn(v1_c1, v2_c1, v3_c1, combo.t1, combo.t2, combo.t3, sign_mask0).simdView;
        simde__m128 r1 = this->_AssembleAdjugateColumn(v1_c0, v2_c0, v3_c0, combo.t1, combo.t2, combo.t3, sign_mask1).simdView;
        simde__m128 r2 = this->_AssembleAdjugateColumn(v1_c3, v2_c3, v3_c3, combo.s1, combo.s2, combo.s3, sign_mask0).simdView;
        simde__m128 r3 = this->_AssembleAdjugateColumn(v1_c2, v2_c2, v3_c2, combo.s1, combo.s2, combo.s3, sign_mask1).simdView;

        // 5. Transposes Cofactor columns in-register to obtain Adjugate columns (C^T)
        _MM_TRANSPOSE4_PS(r0, r1, r2, r3);

        // 6. Stores transposed registers into final column-major Adjugate Matrix
        Mat4x4f adj;
        adj.c0 = r0;
        adj.c1 = r1;
        adj.c2 = r2;
        adj.c3 = r3;

        return adj;
    }
//#endregion

//#region MatrixTransformations
    // Returns a 4x4 Identity Matrix
    static inline Mat4x4f Identity() {
        return Mat4x4f();
    }

    // Returns a 4x4 Zero Matrix (all elements set to 0.0f)
    static inline Mat4x4f nullMatrix() {
        Mat4x4f mat;
        mat.c0 = Vec4f();
        mat.c1 = Vec4f();
        mat.c2 = Vec4f();
        mat.c3 = Vec4f();
        return mat;
    }

    // Builds a 3D translation matrix with the offset stored in Column 3
    static inline Mat4x4f CreateTranslation(const Vec3f &offset) {
        Mat4x4f mat = Mat4x4f();
        mat.c3 = offset.ToVec4f_Point();
        return mat;
    }

    // Builds a non-uniform 3D scaling matrix along the main diagonal
    static inline Mat4x4f CreateScale(const Vec3f& scale) {
        Mat4x4f res = Mat4x4f::Identity();

        res[0][0] = scale.x;
        res[1][1] = scale.y;
        res[2][2] = scale.z;

        return res;
    }

    // Builds a counter-clockwise rotation matrix around the X-axis (pitch, in radians)
    static inline Mat4x4f CreateRotationX(const float angle) {
        Mat4x4f res = Mat4x4f::Identity();
        const float cos = std::cos(angle);
        const float sin = std::sin(angle);

        res[1][1] = cos;
        res[1][2] = sin;

        res[2][1] = -sin;
        res[2][2] = cos;
        
        return res;
    }

    // Builds a counter-clockwise rotation matrix around the Y-axis (yaw, in radians)
    static inline Mat4x4f CreateRotationY(const float angle) {
        Mat4x4f res = Mat4x4f::Identity();
        const float cos = std::cos(angle);
        const float sin = std::sin(angle);

        res[0][0] = cos;
        res[0][2] = -sin;

        res[2][0] = sin;
        res[2][2] = cos;
        
        return res;
    }

    // Builds a counter-clockwise rotation matrix around the Z-axis (roll, in radians)
    static inline Mat4x4f CreateRotationZ(const float angle) {
        Mat4x4f res = Mat4x4f::Identity();
        const float cos = std::cos(angle);
        const float sin = std::sin(angle);

        res[0][0] = cos;
        res[0][1] = sin;

        res[1][0] = -sin;
        res[1][1] = cos;
        
        return res;
    }

    // Builds a combined Euler rotation matrix (Rz * Ry * Rx) from pitch/yaw/roll angles in radians
    static inline Mat4x4f CreateRotationXYZ(const Vec3f& angles) {
        Mat4x4f res;

        const float cosX = std::cos(angles.x);
        const float sinX = std::sin(angles.x);
        const float cosY = std::cos(angles.y);
        const float sinY = std::sin(angles.y);
        const float cosZ = std::cos(angles.z);
        const float sinZ = std::sin(angles.z);

        // Pre-multiplied basis columns for Rz * Ry * Rx
        res.c0 = Vec4f(cosZ * cosY, sinZ * cosY, -sinY, 0.0f);
        res.c1 = Vec4f(cosZ * sinY * sinX - sinZ * cosX, sinZ * sinY * sinX + cosZ * cosX, cosY * sinX, 0.0f);
        res.c2 = Vec4f(cosZ * sinY * cosX + sinZ * sinX, sinZ * sinY * cosX - cosZ * sinX, cosY * cosX, 0.0f);
        res.c3 = Vec4f(0.0f, 0.0f, 0.0f, 1.0f);
        
        return res;
    }

    // Builds a full TRS (Translation * Rotation * Scale) affine model matrix directly
    static inline Mat4x4f CreateLinearTransformation(const Vec3f& offset, const Vec3f& scale, const Vec3f& angles) {
        Mat4x4f res = CreateRotationXYZ(angles);
        // Scales the 3 rotational basis columns directly instead of performing full matrix multiplication
        res.c0 *= scale.x;
        res.c1 *= scale.y;
        res.c2 *= scale.z;
        // Injects world translation into the 4th column
        res.c3 = offset.ToVec4f_Point();
        return res;
    }

    // Builds a Right-Handed Look-At View Matrix (transforms World Space into Camera/View Space)
    static inline Mat4x4f ViewMatrix(const Vec3f& eyePosition, const Vec3f& target, const Vec3f& up) {
        // Camera looks down -Z in right-handed convention, so +Z axis points from target to eye
        const Vec3f f_axis = (eyePosition - target).Normalize();
        // Right axis (+X) orthogonal to world up and camera +Z
        const Vec3f r_axis = up.Cross(f_axis).Normalize();
        // True camera up (+Y); already unit-length since f_axis and r_axis are orthonormal
        const Vec3f u_axis = f_axis.Cross(r_axis);

        // Projects eye position onto orthonormal camera basis to compute negated view translation
        const float dot_r = r_axis.Dot(eyePosition);
        const float dot_u = u_axis.Dot(eyePosition);
        const float dot_f = f_axis.Dot(eyePosition);

        // Packs transposed basis axes into rows 0..2 and projected translation into column 3
        Mat4x4f viewMat = Mat4x4f();
        viewMat.c0 = Vec4f(r_axis.x, u_axis.x, f_axis.x, 0.0f);
        viewMat.c1 = Vec4f(r_axis.y, u_axis.y, f_axis.y, 0.0f);
        viewMat.c2 = Vec4f(r_axis.z, u_axis.z, f_axis.z, 0.0f);
        viewMat.c3 = Vec4f(-dot_r, -dot_u, -dot_f, 1.0f);

        return viewMat;
    }

    // Fast analytical inverse of the View Matrix (transforms Camera Space back to World Space)
    static inline Mat4x4f CameraTransformMatrix(const Mat4x4f& viewMatrix, const Vec3f& eyePosition) {
        // Transposes the entire matrix to invert the upper-left 3x3 orthonormal rotation block
        Mat4x4f inverse = viewMatrix.Transposed();
        // Clears the bottom row (which absorbed the old translation column during transpose)
        inverse[0][3] = 0.0f;
        inverse[1][3] = 0.0f;
        inverse[2][3] = 0.0f;
        inverse[3][3] = 0.0f;
        // Restores camera world position directly into translation column 3
        inverse.c3 = Vec4f(eyePosition.x, eyePosition.y, eyePosition.z, 1.0f);
        return inverse;
    }

    // Builds a Right-Handed Perspective Projection Matrix mapping frustum to NDC [-1, 1]
    static inline Mat4x4f PerspectiveProjectionMatrix(float tanHalfFov, float aspect_ratio, float z_near, float z_far) {
        Mat4x4f perspectiveMat = Mat4x4f::nullMatrix(); 
        const float nearMinusFar = z_near - z_far;

        // X scale: maps horizontal FOV to [-1, 1] accounting for viewport aspect ratio
        perspectiveMat[0][0] = 1.0f / (aspect_ratio * tanHalfFov);
        
        // Y scale: maps vertical FOV to [-1, 1]
        perspectiveMat[1][1] = 1.0f / tanHalfFov;
        
        // Z remapping coefficients and w_clip = -z_view perspective divide trigger
        perspectiveMat[2][2] = (z_far + z_near) / nearMinusFar;
        perspectiveMat[2][3] = -1.0f;
        
        // Depth offset term for non-linear Z-buffer mapping
        perspectiveMat[3][2] = (2 * z_far * z_near) / nearMinusFar;

        return perspectiveMat;
    }

    // Fast analytical inverse of the Perspective Projection Matrix (unprojects Clip Space to View Space)
    static Mat4x4f PerspectiveMatrixInverse(const Mat4x4f& perspectiveMatrix, const float z_near, const float z_far) {
        Mat4x4f inverse = Mat4x4f::nullMatrix();

        // Inverts X and Y focal scaling factors
        inverse.c0.x = 1.0f / perspectiveMatrix.c0.x;
        inverse.c1.y = 1.0f / perspectiveMatrix.c1.y;

        // Analytically inverts the sparse 2x2 Z/W projection sub-block
        const float D = (2 * z_far * z_near);

        inverse[2][3] = (z_near - z_far) / D;
        inverse[3][2] = -1.0f;
        inverse[3][3] = (z_far + z_near) / D;

        return inverse;
    }
//#endregion

//#region IndexingOperators
    // Returns column vector at index (read-only)
    inline Vec4f MATH_VECTORCALL operator[](const size_t index) const {
        assert(index < 4);
        return this->cols[index];
    }

    // Returns mutable reference to column vector at index
    inline Vec4f& MATH_VECTORCALL operator[](const size_t index) {
        assert(index < 4);
        return this->cols[index];
    }
//#endregion
};