#include <gtest/gtest.h>
#include <cmath>
#include <rasterizer/math_core/Primitives.hpp>

TEST(Vec4fTest, ConstructorsAndMemoryLayout) {
    // Default Constructor
    Vec4f vZero;
    EXPECT_FLOAT_EQ(vZero.x, 0.0f);
    EXPECT_FLOAT_EQ(vZero.y, 0.0f);
    EXPECT_FLOAT_EQ(vZero.z, 0.0f);
    EXPECT_FLOAT_EQ(vZero.w, 0.0f);

    // Scalar Constructor
    Vec4f vFull(1.0f, 2.0f, 3.0f, 4.0f);
    EXPECT_FLOAT_EQ(vFull.x, 1.0f);
    EXPECT_FLOAT_EQ(vFull.y, 2.0f);
    EXPECT_FLOAT_EQ(vFull.z, 3.0f);
    EXPECT_FLOAT_EQ(vFull.w, 4.0f);

    // Union Aliasing Verification
    EXPECT_FLOAT_EQ(vFull.arrView[0], 1.0f);
    EXPECT_FLOAT_EQ(vFull.arrView[1], 2.0f);
    EXPECT_FLOAT_EQ(vFull.arrView[2], 3.0f);
    EXPECT_FLOAT_EQ(vFull.arrView[3], 4.0f);

    // Broadcast Constructor
    Vec4f vBroad(5.5f);
    EXPECT_FLOAT_EQ(vBroad.x, 5.5f);
    EXPECT_FLOAT_EQ(vBroad.y, 5.5f);
    EXPECT_FLOAT_EQ(vBroad.z, 5.5f);
    EXPECT_FLOAT_EQ(vBroad.w, 5.5f);

    // Pointer Constructor with aligned SIMD load
    alignas(16) float rawData[4] = { 10.0f, 20.0f, 30.0f, 40.0f };
    Vec4f vPtr(rawData);
    EXPECT_FLOAT_EQ(vPtr.x, 10.0f);
    EXPECT_FLOAT_EQ(vPtr.y, 20.0f);
    EXPECT_FLOAT_EQ(vPtr.z, 30.0f);
    EXPECT_FLOAT_EQ(vPtr.w, 40.0f);
}

TEST(Vec4fTest, IndexingOperators) {
    Vec4f v(1.0f, 2.0f, 3.0f, 4.0f);

    // Const Read
    const Vec4f& vConst = v;
    EXPECT_FLOAT_EQ(vConst[0], 1.0f);
    EXPECT_FLOAT_EQ(vConst[1], 2.0f);
    EXPECT_FLOAT_EQ(vConst[2], 3.0f);
    EXPECT_FLOAT_EQ(vConst[3], 4.0f);

    // Non-Const Write
    v[0] = 100.0f;
    v[3] = 400.0f;
    EXPECT_FLOAT_EQ(v.x, 100.0f);
    EXPECT_FLOAT_EQ(v.w, 400.0f);
}

TEST(Vec4fTest, VectorArithmeticOperators) {
    Vec4f a(1.0f, 2.0f, 3.0f, 4.0f);
    Vec4f b(5.0f, 6.0f, 7.0f, 8.0f);

    // Vector Sum
    Vec4f sum = a + b;
    EXPECT_FLOAT_EQ(sum.x, 6.0f);
    EXPECT_FLOAT_EQ(sum.y, 8.0f);
    EXPECT_FLOAT_EQ(sum.z, 10.0f);
    EXPECT_FLOAT_EQ(sum.w, 12.0f);

    // Vector Sub
    Vec4f sub = b - a;
    EXPECT_FLOAT_EQ(sub.x, 4.0f);
    EXPECT_FLOAT_EQ(sub.y, 4.0f);
    EXPECT_FLOAT_EQ(sub.z, 4.0f);
    EXPECT_FLOAT_EQ(sub.w, 4.0f);

    // Vector Complement
    Vec4f neg = -a;
    EXPECT_FLOAT_EQ(neg.x, -1.0f);
    EXPECT_FLOAT_EQ(neg.y, -2.0f);
    EXPECT_FLOAT_EQ(neg.z, -3.0f);
    EXPECT_FLOAT_EQ(neg.w, -4.0f);

    // Hadamard Product
    Vec4f hadamard = a * b;
    EXPECT_FLOAT_EQ(hadamard.x, 5.0f);   // 1 * 5
    EXPECT_FLOAT_EQ(hadamard.y, 12.0f);  // 2 * 6
    EXPECT_FLOAT_EQ(hadamard.z, 21.0f);  // 3 * 7
    EXPECT_FLOAT_EQ(hadamard.w, 32.0f);  // 4 * 8

    // Vector Sum, with assigement
    Vec4f accum(1.0f, 1.0f, 1.0f, 1.0f);
    accum += a;
    EXPECT_FLOAT_EQ(accum.x, 2.0f);
    EXPECT_FLOAT_EQ(accum.y, 3.0f);
    EXPECT_FLOAT_EQ(accum.z, 4.0f);
    EXPECT_FLOAT_EQ(accum.w, 5.0f);
}

TEST(Vec4fTest, ScalarArithmeticOperators) {
    Vec4f v(2.0f, -4.0f, 6.0f, -8.0f);

    // Vector * Scalar
    Vec4f scaled = v * 2.5f;
    EXPECT_FLOAT_EQ(scaled.x, 5.0f);
    EXPECT_FLOAT_EQ(scaled.y, -10.0f);
    EXPECT_FLOAT_EQ(scaled.z, 15.0f);
    EXPECT_FLOAT_EQ(scaled.w, -20.0f);

    // Vector / Scalar
    Vec4f div = v / 2.0f;
    EXPECT_FLOAT_EQ(div.x, 1.0f);
    EXPECT_FLOAT_EQ(div.y, -2.0f);
    EXPECT_FLOAT_EQ(div.z, 3.0f);
    EXPECT_FLOAT_EQ(div.w, -4.0f);

    // Vector / Scalar, with assigement
    v /= 2.0f;
    EXPECT_FLOAT_EQ(v.x, 1.0f);
    EXPECT_FLOAT_EQ(v.y, -2.0f);
    EXPECT_FLOAT_EQ(v.z, 3.0f);
    EXPECT_FLOAT_EQ(v.w, -4.0f);
}

TEST(Vec4fTest, ComparisonOperators) {
    Vec4f a(1.0f, 2.0f, 3.0f, 4.0f);
    Vec4f aCopy(1.0f, 2.0f, 3.0f, 4.0f);
    Vec4f b(1.0f, 2.0f, 3.0f, 4.001f);

    // Exact Comparison
    EXPECT_TRUE(a == aCopy);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);

    // Tollerant Comparison
    Vec4f vBase(1.0f, 2.0f, 3.0f, 4.0f);
    Vec4f vNear(1.000001f, 1.999999f, 3.000002f, 3.999998f);
    EXPECT_TRUE(vBase.IsNearlyEqual(vNear, 1e-4f));
    EXPECT_FALSE(vBase.IsNearlyEqual(b, 1e-4f));
}

TEST(Vec4fTest, GeometricOperations) {
    Vec4f v1(1.0f, 2.0f, 3.0f, 4.0f);
    Vec4f v2(2.0f, 3.0f, 4.0f, 5.0f);

    // Dot Product 4D: (1*2 + 2*3 + 3*4 + 4*5) = 2 + 6 + 12 + 20 = 40
    float dot4D = v1.Dot(v2);
    EXPECT_FLOAT_EQ(dot4D, 40.0f);

    // Dot Product 3D: (1*2 + 2*3 + 3*4) = 2 + 6 + 12 = 20
    float dot3D = v1.Dot3(v2);
    EXPECT_FLOAT_EQ(dot3D, 20.0f);

    // Length Squared: 1^2 + 2^2 + 3^2 + 4^2 = 1 + 4 + 9 + 16 = 30
    EXPECT_FLOAT_EQ(v1.SquaredMagnitude(), 30.0f);

    // Length: sqrt(30) ~= 5.477225575
    EXPECT_FLOAT_EQ(v1.Magnitude(), std::sqrt(30.0f));

    // Normalization
    Vec4f vNorm = v1.Normalize();
    EXPECT_NEAR(vNorm.Magnitude(), 1.0f, 1e-3f);

    // Verifies if the length is preserved
    Vec4f vReconstructed = vNorm * v1.Magnitude();
    EXPECT_TRUE(v1.IsNearlyEqual(vReconstructed, 1e-3f));
}

TEST(Vec4fTest, ConversionOperators) {
    Vec4f v1(1.0f, 2.0f, 3.0f, 4.0f);

    // Cast to Vec2f
    Vec2f v1ToVec2f = v1.ToVec2f();
    EXPECT_FLOAT_EQ(v1ToVec2f.x, v1.x);
    EXPECT_FLOAT_EQ(v1ToVec2f.y, v1.y);
    // Cast to Vec3f
    Vec3f v1ToVec3f = v1.ToVec3f();
    EXPECT_FLOAT_EQ(v1ToVec3f.x, v1.x);
    EXPECT_FLOAT_EQ(v1ToVec3f.y, v1.y);
    EXPECT_FLOAT_EQ(v1ToVec3f.z, v1.z);
}