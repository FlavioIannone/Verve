#include <gtest/gtest.h>
#include <cmath>
#include <rasterizer/math_core/Primitives.hpp>

TEST(Vec3fTest, ConstructorsAndMemoryLayout) {
    // Default constructor (initializes to zero)
    Vec3f vZero;
    EXPECT_FLOAT_EQ(vZero.x, 0.0f);
    EXPECT_FLOAT_EQ(vZero.y, 0.0f);
    EXPECT_FLOAT_EQ(vZero.z, 0.0f);

    // Component-wise constructor
    Vec3f vFull(1.0f, 2.0f, 3.0f);
    EXPECT_FLOAT_EQ(vFull.x, 1.0f);
    EXPECT_FLOAT_EQ(vFull.y, 2.0f);
    EXPECT_FLOAT_EQ(vFull.z, 3.0f);

    // Union array aliasing verification
    EXPECT_FLOAT_EQ(vFull.arrView[0], 1.0f);
    EXPECT_FLOAT_EQ(vFull.arrView[1], 2.0f);
    EXPECT_FLOAT_EQ(vFull.arrView[2], 3.0f);

    // Broadcast constructor
    Vec3f vBroad(7.5f);
    EXPECT_FLOAT_EQ(vBroad.x, 7.5f);
    EXPECT_FLOAT_EQ(vBroad.y, 7.5f);
    EXPECT_FLOAT_EQ(vBroad.z, 7.5f);

    // Pointer-based constructor
    float rawData[3] = { 10.0f, 20.0f, 30.0f };
    Vec3f vPtr(rawData);
    EXPECT_FLOAT_EQ(vPtr.x, 10.0f);
    EXPECT_FLOAT_EQ(vPtr.y, 20.0f);
    EXPECT_FLOAT_EQ(vPtr.z, 30.0f);
}

TEST(Vec3fTest, IndexingOperators) {
    Vec3f v(1.0f, 2.0f, 3.0f);

    // Const read indexing
    const Vec3f& vConst = v;
    EXPECT_FLOAT_EQ(vConst[0], 1.0f);
    EXPECT_FLOAT_EQ(vConst[1], 2.0f);
    EXPECT_FLOAT_EQ(vConst[2], 3.0f);

    // Non-const write indexing
    v[0] = 100.0f;
    v[2] = 300.0f;
    EXPECT_FLOAT_EQ(v.x, 100.0f);
    EXPECT_FLOAT_EQ(v.z, 300.0f);
}

TEST(Vec3fTest, VectorArithmeticOperators) {
    Vec3f a(1.0f, 2.0f, 3.0f);
    Vec3f b(4.0f, 5.0f, 6.0f);

    // Vector addition
    Vec3f sum = a + b;
    EXPECT_FLOAT_EQ(sum.x, 5.0f);
    EXPECT_FLOAT_EQ(sum.y, 7.0f);
    EXPECT_FLOAT_EQ(sum.z, 9.0f);

    // Vector subtraction
    Vec3f sub = b - a;
    EXPECT_FLOAT_EQ(sub.x, 3.0f);
    EXPECT_FLOAT_EQ(sub.y, 3.0f);
    EXPECT_FLOAT_EQ(sub.z, 3.0f);

    // Vector negation
    Vec3f neg = -a;
    EXPECT_FLOAT_EQ(neg.x, -1.0f);
    EXPECT_FLOAT_EQ(neg.y, -2.0f);
    EXPECT_FLOAT_EQ(neg.z, -3.0f);

    // Component-wise Hadamard product
    Vec3f hadamard = a * b;
    EXPECT_FLOAT_EQ(hadamard.x, 4.0f);
    EXPECT_FLOAT_EQ(hadamard.y, 10.0f);
    EXPECT_FLOAT_EQ(hadamard.z, 18.0f);

    // Compound assignment addition
    Vec3f accum(1.0f, 1.0f, 1.0f);
    accum += a;
    EXPECT_FLOAT_EQ(accum.x, 2.0f);
    EXPECT_FLOAT_EQ(accum.y, 3.0f);
    EXPECT_FLOAT_EQ(accum.z, 4.0f);
}

TEST(Vec3fTest, ScalarArithmeticOperators) {
    Vec3f v(2.0f, -4.0f, 6.0f);

    // Vector * Scalar
    Vec3f scaled = v * 2.5f;
    EXPECT_FLOAT_EQ(scaled.x, 5.0f);
    EXPECT_FLOAT_EQ(scaled.y, -10.0f);
    EXPECT_FLOAT_EQ(scaled.z, 15.0f);

    // Vector / Scalar
    Vec3f div = v / 2.0f;
    EXPECT_FLOAT_EQ(div.x, 1.0f);
    EXPECT_FLOAT_EQ(div.y, -2.0f);
    EXPECT_FLOAT_EQ(div.z, 3.0f);

    // Compound assignment division
    v /= 2.0f;
    EXPECT_FLOAT_EQ(v.x, 1.0f);
    EXPECT_FLOAT_EQ(v.y, -2.0f);
    EXPECT_FLOAT_EQ(v.z, 3.0f);
}

TEST(Vec3fTest, ComparisonOperators) {
    Vec3f a(1.0f, 2.0f, 3.0f);
    Vec3f aCopy(1.0f, 2.0f, 3.0f);
    Vec3f b(1.0f, 2.0f, 3.001f);

    // Exact equality and inequality
    EXPECT_TRUE(a == aCopy);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);

    // Tolerant floating-point equality
    Vec3f vBase(1.0f, 2.0f, 3.0f);
    Vec3f vNear(1.00001f, 1.99999f, 3.00001f);
    EXPECT_TRUE(vBase.IsNearlyEqual(vNear));
    EXPECT_FALSE(vBase.IsNearlyEqual(b));
}

TEST(Vec3fTest, GeometricOperations) {
    Vec3f v1(1.0f, 2.0f, 3.0f);
    Vec3f v2(2.0f, 3.0f, 4.0f);

    // Dot product: (1*2 + 2*3 + 3*4) = 2 + 6 + 12 = 20
    EXPECT_FLOAT_EQ(v1.Dot(v2), 20.0f);

    // Cross product: (2*4 - 3*3, 3*2 - 1*4, 1*3 - 2*2) = (-1, 2, -1)
    Vec3f cross = v1.Cross(v2);
    EXPECT_FLOAT_EQ(cross.x, -1.0f);
    EXPECT_FLOAT_EQ(cross.y, 2.0f);
    EXPECT_FLOAT_EQ(cross.z, -1.0f);

    // Cross product properties: Cross(v, v) must be null
    Vec3f selfCross = v1.Cross(v1);
    EXPECT_FLOAT_EQ(selfCross.x, 0.0f);
    EXPECT_FLOAT_EQ(selfCross.y, 0.0f);
    EXPECT_FLOAT_EQ(selfCross.z, 0.0f);

    // Length and Length Squared: 1 + 4 + 9 = 14
    EXPECT_FLOAT_EQ(v1.SquaredMagnitude(), 14.0f);
    EXPECT_FLOAT_EQ(v1.Magnitude(), std::sqrt(14.0f));

    // Vector normalization
    Vec3f vNorm = v1.Normalized();
    EXPECT_NEAR(vNorm.Magnitude(), 1.0f, 1e-4f);

    // Reconstruct vector from normalized direction and magnitude
    Vec3f reconstructed = vNorm * v1.Magnitude();
    EXPECT_TRUE(v1.IsNearlyEqual(reconstructed, 1e-4f));
}

TEST(Vec3fTest, ConversionOperators) {
    Vec3f v(1.0f, 2.0f, 3.0f);

    // Conversion to Vec2f
    Vec2f v2 = v.ToVec2f();
    EXPECT_FLOAT_EQ(v2.x, 1.0f);
    EXPECT_FLOAT_EQ(v2.y, 2.0f);

    // Conversion to Vec4f with arbitrary w component
    Vec4f v4Arbitrary = v.ToVec4f(5.0f);
    EXPECT_FLOAT_EQ(v4Arbitrary.x, 1.0f);
    EXPECT_FLOAT_EQ(v4Arbitrary.y, 2.0f);
    EXPECT_FLOAT_EQ(v4Arbitrary.z, 3.0f);
    EXPECT_FLOAT_EQ(v4Arbitrary.w, 5.0f);

    // Homogeneous Point conversion (w = 1.0)
    Vec4f v4Point = v.ToVec4f_Point();
    EXPECT_FLOAT_EQ(v4Point.x, 1.0f);
    EXPECT_FLOAT_EQ(v4Point.y, 2.0f);
    EXPECT_FLOAT_EQ(v4Point.z, 3.0f);
    EXPECT_FLOAT_EQ(v4Point.w, 1.0f);

    // Homogeneous Direction conversion (w = 0.0)
    Vec4f v4Dir = v.ToVec4f_Direction();
    EXPECT_FLOAT_EQ(v4Dir.x, 1.0f);
    EXPECT_FLOAT_EQ(v4Dir.y, 2.0f);
    EXPECT_FLOAT_EQ(v4Dir.z, 3.0f);
    EXPECT_FLOAT_EQ(v4Dir.w, 0.0f);
}