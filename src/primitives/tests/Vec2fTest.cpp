#include <gtest/gtest.h>
#include <rasterizer/math_core/Primitives.hpp>

TEST(Vec2fTest, ConstructorsAndMemoryLayout) {
    // Default constructor (zeros)
    Vec2f vZero;
    EXPECT_FLOAT_EQ(vZero.x, 0.0f);
    EXPECT_FLOAT_EQ(vZero.y, 0.0f);

    // Component-wise constructor
    Vec2f vFull(3.0f, 4.0f);
    EXPECT_FLOAT_EQ(vFull.x, 3.0f);
    EXPECT_FLOAT_EQ(vFull.y, 4.0f);

    // Union array aliasing verification
    EXPECT_FLOAT_EQ(vFull.arrView[0], 3.0f);
    EXPECT_FLOAT_EQ(vFull.arrView[1], 4.0f);

    // Broadcast constructor
    Vec2f vBroad(8.5f);
    EXPECT_FLOAT_EQ(vBroad.x, 8.5f);
    EXPECT_FLOAT_EQ(vBroad.y, 8.5f);

    // Raw array constructor
    float rawData[2] = { 12.0f, 24.0f };
    Vec2f vPtr(rawData);
    EXPECT_FLOAT_EQ(vPtr.x, 12.0f);
    EXPECT_FLOAT_EQ(vPtr.y, 24.0f);
}

TEST(Vec2fTest, IndexingOperators) {
    Vec2f v(5.0f, 10.0f);

    // Const read indexing
    const Vec2f& vConst = v;
    EXPECT_FLOAT_EQ(vConst[0], 5.0f);
    EXPECT_FLOAT_EQ(vConst[1], 10.0f);

    // Non-const write indexing
    v[0] = 50.0f;
    v[1] = 100.0f;
    EXPECT_FLOAT_EQ(v.x, 50.0f);
    EXPECT_FLOAT_EQ(v.y, 100.0f);
}

TEST(Vec2fTest, ArithmeticOperators) {
    Vec2f a(2.0f, 3.0f);
    Vec2f b(4.0f, 5.0f);

    // Vector addition
    Vec2f sum = a + b;
    EXPECT_FLOAT_EQ(sum.x, 6.0f);
    EXPECT_FLOAT_EQ(sum.y, 8.0f);

    // Vector subtraction
    Vec2f sub = b - a;
    EXPECT_FLOAT_EQ(sub.x, 2.0f);
    EXPECT_FLOAT_EQ(sub.y, 2.0f);

    // Vector negation
    Vec2f neg = -a;
    EXPECT_FLOAT_EQ(neg.x, -2.0f);
    EXPECT_FLOAT_EQ(neg.y, -3.0f);

    // Hadamard product
    Vec2f hadamard = a * b;
    EXPECT_FLOAT_EQ(hadamard.x, 8.0f);
    EXPECT_FLOAT_EQ(hadamard.y, 15.0f);

    // Scalar multiplication and division
    Vec2f scaled = a * 3.0f;
    EXPECT_FLOAT_EQ(scaled.x, 6.0f);
    EXPECT_FLOAT_EQ(scaled.y, 9.0f);

    Vec2f div = b / 2.0f;
    EXPECT_FLOAT_EQ(div.x, 2.0f);
    EXPECT_FLOAT_EQ(div.y, 2.5f);
}

TEST(Vec2fTest, ComparisonOperators) {
    Vec2f a(1.5f, 2.5f);
    Vec2f aCopy(1.5f, 2.5f);
    Vec2f b(1.5f, 2.501f);

    EXPECT_TRUE(a == aCopy);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);

    Vec2f vNear(1.50001f, 2.49999f);
    EXPECT_TRUE(a.IsNearlyEqual(vNear, 1e-4f));
    EXPECT_FALSE(a.IsNearlyEqual(b, 1e-4f));
}

TEST(Vec2fTest, GeometricOperations) {
    Vec2f v1(3.0f, 4.0f);
    Vec2f v2(1.0f, 2.0f);

    // Dot product: (3*1 + 4*2) = 11
    EXPECT_FLOAT_EQ(v1.Dot(v2), 11.0f);

    // Length and Length Squared: 3^2 + 4^2 = 25
    EXPECT_FLOAT_EQ(v1.SquaredMagnitude(), 25.0f);
    EXPECT_FLOAT_EQ(v1.Magnitude(), 5.0f);

    // Normalization
    Vec2f vNorm = v1.Normalized();
    EXPECT_FLOAT_EQ(vNorm.x, 0.6f);
    EXPECT_FLOAT_EQ(vNorm.y, 0.8f);
    EXPECT_NEAR(vNorm.Magnitude(), 1.0f, 1e-5f);
}

TEST(Vec2fTest, ConversionOperators) {
    Vec2f v(7.0f, 9.0f);

    // Conversion to Vec3f with specified z
    Vec3f v3 = v.ToVec3f(11.0f);
    EXPECT_FLOAT_EQ(v3.x, 7.0f);
    EXPECT_FLOAT_EQ(v3.y, 9.0f);
    EXPECT_FLOAT_EQ(v3.z, 11.0f);

    // Conversion to Vec4f with specified z and w
    Vec4f v4 = v.ToVec4f(11.0f, 13.0f);
    EXPECT_FLOAT_EQ(v4.x, 7.0f);
    EXPECT_FLOAT_EQ(v4.y, 9.0f);
    EXPECT_FLOAT_EQ(v4.z, 11.0f);
    EXPECT_FLOAT_EQ(v4.w, 13.0f);
}