#include <gtest/gtest.h>
#include <rasterizer/math_core/Primitives.hpp>


TEST(Mat4x4fTest, ConstructorsAndMemoryLayout) {
    // Null matrix (all zeros)
    Mat4x4f mNull = Mat4x4f::nullMatrix();
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            EXPECT_FLOAT_EQ(mNull[col][row], 0.0f);
        }
    }

    // Identity matrix
    Mat4x4f mIdent = Mat4x4f::Identity();
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            if (col == row) {
                EXPECT_FLOAT_EQ(mIdent[col][row], 1.0f);
            } else {
                EXPECT_FLOAT_EQ(mIdent[col][row], 0.0f);
            }
        }
    }

    // Column vectors assignment (Column-Major layout)
    Mat4x4f mCols;
    mCols.c0 = Vec4f(1.0f, 2.0f, 3.0f, 4.0f);
    mCols.c1 = Vec4f(5.0f, 6.0f, 7.0f, 8.0f);
    mCols.c2 = Vec4f(9.0f, 10.0f, 11.0f, 12.0f);
    mCols.c3 = Vec4f(13.0f, 14.0f, 15.0f, 16.0f);

    // Verify indexing: mat[col][row]
    EXPECT_FLOAT_EQ(mCols[0][0], 1.0f);
    EXPECT_FLOAT_EQ(mCols[0][3], 4.0f);
    EXPECT_FLOAT_EQ(mCols[3][0], 13.0f);
    EXPECT_FLOAT_EQ(mCols[3][3], 16.0f);
}

TEST(Mat4x4fTest, MatrixVectorMultiplication) {
    // Identity * Vector = Vector
    Mat4x4f ident = Mat4x4f::Identity();
    Vec4f v(2.0f, 4.0f, 6.0f, 1.0f);
    Vec4f vRes = ident * v;
    EXPECT_TRUE(v.IsNearlyEqual(vRes, 1e-5f));

    // Translation matrix * Point
    Vec3f translationOffset(10.0f, -5.0f, 2.0f);
    Mat4x4f transMat = Mat4x4f::CreateTranslation(translationOffset);

    // Point (w = 1.0) must translate
    Vec4f point(1.0f, 2.0f, 3.0f, 1.0f);
    Vec4f translatedPoint = transMat * point;
    EXPECT_FLOAT_EQ(translatedPoint.x, 11.0f);
    EXPECT_FLOAT_EQ(translatedPoint.y, -3.0f);
    EXPECT_FLOAT_EQ(translatedPoint.z, 5.0f);
    EXPECT_FLOAT_EQ(translatedPoint.w, 1.0f);

    // Direction (w = 0.0) must ignore translation
    Vec4f dir(1.0f, 2.0f, 3.0f, 0.0f);
    Vec4f translatedDir = transMat * dir;
    EXPECT_FLOAT_EQ(translatedDir.x, 1.0f);
    EXPECT_FLOAT_EQ(translatedDir.y, 2.0f);
    EXPECT_FLOAT_EQ(translatedDir.z, 3.0f);
    EXPECT_FLOAT_EQ(translatedDir.w, 0.0f);
}

TEST(Mat4x4fTest, MatrixMatrixMultiplication) {
    // Scale matrix * Translation matrix
    Mat4x4f s = Mat4x4f::CreateScale(Vec3f(2.0f, 3.0f, 4.0f));
    Mat4x4f t = Mat4x4f::CreateTranslation(Vec3f(5.0f, 5.0f, 5.0f));

    // In Column-Major: (T * S) scales first, then translates
    Mat4x4f ts = t * s;
    Vec4f point(1.0f, 1.0f, 1.0f, 1.0f);
    Vec4f transformed = ts * point;

    // (1 * 2) + 5 = 7, (1 * 3) + 5 = 8, (1 * 4) + 5 = 9
    EXPECT_FLOAT_EQ(transformed.x, 7.0f);
    EXPECT_FLOAT_EQ(transformed.y, 8.0f);
    EXPECT_FLOAT_EQ(transformed.z, 9.0f);
    EXPECT_FLOAT_EQ(transformed.w, 1.0f);
}

TEST(Mat4x4fTest, Transposition) {
    Mat4x4f m = Mat4x4f::nullMatrix();
    m[1][0] = 5.0f; // col 1, row 0
    m[0][1] = 2.0f; // col 0, row 1
    m[3][2] = 9.0f; // col 3, row 2

    Mat4x4f t = m.Transposed();
    EXPECT_FLOAT_EQ(t[0][1], 5.0f);
    EXPECT_FLOAT_EQ(t[1][0], 2.0f);
    EXPECT_FLOAT_EQ(t[2][3], 9.0f);

    // Double transposition preserves the matrix
    Mat4x4f t2 = t.Transposed();
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            EXPECT_FLOAT_EQ(t2[c][r], m[c][r]);
        }
    }
}

TEST(Mat4x4fTest, GeneralInverseAndDeterminant) {
    // Arbitrary non-singular affine matrix
    Mat4x4f m = Mat4x4f::CreateLinearTransformation(
        Vec3f(3.0f, -4.0f, 5.0f),
        Vec3f(2.0f, 1.5f, 0.5f),
        Vec3f(0.3f, -0.2f, 0.8f)
    );

    float det = m.DeterminantGeneral();
    EXPECT_NE(det, 0.0f);

    Mat4x4f inv;
    EXPECT_TRUE(m.InverseGeneral(inv));
}

TEST(Mat4x4fTest, ViewAndCameraTransformAnalyticalInverse) {
    Vec3f eye(0.0f, 5.0f, 10.0f);
    Vec3f target(0.0f, 0.0f, 0.0f);
    Vec3f up(0.0f, 1.0f, 0.0f);

    // Compute ViewMatrix and CameraTransformMatrix
    Mat4x4f view = Mat4x4f::ViewMatrix(eye, target, up);
    Mat4x4f camWorld = Mat4x4f::CameraTransformMatrix(view, eye);

    // Verify analytical inverse: View * CamWorld = Identity
    Mat4x4f identCheck = view * camWorld;
    Mat4x4f expectedIdent = Mat4x4f::Identity();

    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            EXPECT_NEAR(identCheck[c][r], expectedIdent[c][r], 1e-4f);
        }
    }

    // Verify that eye position maps to origin (0, 0, 0, 1) in Eye Space
    Vec4f eyeHomogeneous = eye.ToVec4f_Point();
    Vec4f viewSpaceEye = view * eyeHomogeneous;
    EXPECT_NEAR(viewSpaceEye.x, 0.0f, 1e-4f);
    EXPECT_NEAR(viewSpaceEye.y, 0.0f, 1e-4f);
    EXPECT_NEAR(viewSpaceEye.z, 0.0f, 1e-4f);
    EXPECT_FLOAT_EQ(viewSpaceEye.w, 1.0f);
}

TEST(Mat4x4fTest, PerspectiveAndPerspectiveInverse) {
    const float fovY = 1.04719755f; // 60 degrees in radians
    const float aspect = 16.0f / 9.0f;
    const float zNear = 0.1f;
    const float zFar = 1000.0f;

    Mat4x4f proj = Mat4x4f::PerspectiveProjectionMatrix(fovY, aspect, zNear, zFar);
    Mat4x4f projInv = Mat4x4f::PerspectiveMatrixInverse(proj, zNear, zFar);

    // Verify analytical inverse: Proj * ProjInv = Identity
    Mat4x4f identCheck = proj * projInv;
    Mat4x4f expectedIdent = Mat4x4f::Identity();

    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            EXPECT_NEAR(identCheck[c][r], expectedIdent[c][r], 1e-4f);
        }
    }
}