#include <gtest/gtest.h>
#include <rasterizer/math_core/Align.hpp>
#include <rasterizer/math_core/Primitives.hpp>

// Vec4f alignement test (expected 16 bytes)
TEST(ALIGNEMENT_TEST, Vec4fAlignment) {
    EXPECT_EQ(alignof(Vec4f), ALIGN_VEC4F);
}

// Vec3f alignement test (expected 12 bytes)
TEST(ALIGNEMENT_TEST, Vec3fAlignment) {
    EXPECT_EQ(alignof(Vec3f), ALIGN_VEC3F);
}

// Vec2f alignement test (expected 8 bytes)
TEST(ALIGNEMENT_TEST, Vec2fAlignment) {
    EXPECT_EQ(alignof(Vec2f), ALIGN_VEC2F);
}

// Mat4x4f alignement test (expected 64 bytes)
TEST(ALIGNEMENT_TEST, Mat4x4fAlignment) {
    EXPECT_EQ(alignof(Mat4x4f), ALIGN_VEC4F * 4); // Four column vectors
}

// Quaternion alignement test (expected 16 bytes)
TEST(ALIGNEMENT_TEST, QuaternionAlignment) {
    EXPECT_EQ(alignof(Quaternion), ALIGN_QUATERNION);
}