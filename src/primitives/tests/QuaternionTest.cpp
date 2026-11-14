#include <gtest/gtest.h>
#include <cmath>
#include <numbers>
#include "rasterizer/primitives/Quaternion.hpp"
#include "rasterizer/primitives/Mat4x4f.hpp"
#include "rasterizer/primitives/Vec3f.hpp"
#include "rasterizer/primitives/Vec4f.hpp"

namespace {
    constexpr float kPi      = std::numbers::pi_v<float>;
    constexpr float kHalfPi  = kPi * 0.5f;
    constexpr float kEps     = 1e-4f;
    constexpr float kTightEps = 1e-5f;
}


TEST(QuaternionTest, MemoryLayoutAndAlignment) {
    // Verifies 16-byte SIMD alignment and exact 16-byte union footprint
    EXPECT_EQ(sizeof(Quaternion), 16u);
    EXPECT_GE(alignof(Quaternion), 16u);

    Quaternion q(1.0f, 2.0f, 3.0f, 4.0f);

    // Checks that scalar members x, y, z, w map to distinct 4-byte lanes
    EXPECT_FLOAT_EQ(q.x, 1.0f);
    EXPECT_FLOAT_EQ(q.y, 2.0f);
    EXPECT_FLOAT_EQ(q.z, 3.0f);
    EXPECT_FLOAT_EQ(q.w, 4.0f);

    // Checks that nested Vec3f 'v' aliases bytes 0..11 [x, y, z] without corrupting 'w'
    EXPECT_FLOAT_EQ(q.v.x, 1.0f);
    EXPECT_FLOAT_EQ(q.v.y, 2.0f);
    EXPECT_FLOAT_EQ(q.v.z, 3.0f);

    q.v = Vec3f(10.0f, 20.0f, 30.0f);
    EXPECT_FLOAT_EQ(q.x, 10.0f);
    EXPECT_FLOAT_EQ(q.y, 20.0f);
    EXPECT_FLOAT_EQ(q.z, 30.0f);
    EXPECT_FLOAT_EQ(q.w, 4.0f); // w must remain untouched at byte offset 12

    // Checks indexing operator against scalar lanes
    EXPECT_FLOAT_EQ(q[0], 10.0f);
    EXPECT_FLOAT_EQ(q[1], 20.0f);
    EXPECT_FLOAT_EQ(q[2], 30.0f);
    EXPECT_FLOAT_EQ(q[3], 4.0f);
}

TEST(QuaternionTest, ConstructorsAndIdentity) {
    // Default constructor and Identity() factory must yield [0, 0, 0, 1]
    Quaternion idDefault;
    Quaternion idFactory = Quaternion::Identity();

    EXPECT_EQ(idDefault, Quaternion(0.0f, 0.0f, 0.0f, 1.0f));
    EXPECT_EQ(idFactory, idDefault);

    // Scalar + Vec3f constructor: Quaternion(w, v) -> [v.x, v.y, v.z, w]
    Vec3f imaginary(2.0f, -3.0f, 5.0f);
    Quaternion fromVec(7.0f, imaginary);
    EXPECT_EQ(fromVec, Quaternion(2.0f, -3.0f, 5.0f, 7.0f));

    // Direct SIMD register constructor
    __m128 rawSimd = _mm_setr_ps(-1.0f, 0.5f, 2.25f, 0.75f);
    Quaternion fromSimd(rawSimd);
    EXPECT_EQ(fromSimd, Quaternion(-1.0f, 0.5f, 2.25f, 0.75f));
}

TEST(QuaternionTest, ComponentWiseArithmeticAndDotProduct) {
    Quaternion a(1.0f, 2.0f, 3.0f, 4.0f);
    Quaternion b(4.0f, -1.0f, 2.0f, 0.5f);

    EXPECT_EQ(a + b, Quaternion(5.0f, 1.0f, 5.0f, 4.5f));
    EXPECT_EQ(a - b, Quaternion(-3.0f, 3.0f, 1.0f, 3.5f));
    EXPECT_EQ(-a, Quaternion(-1.0f, -2.0f, -3.0f, -4.0f));
    EXPECT_EQ(a * 2.0f, Quaternion(2.0f, 4.0f, 6.0f, 8.0f));
    EXPECT_TRUE((a / 2.0f).IsNearlyEqual(Quaternion(0.5f, 1.0f, 1.5f, 2.0f), kTightEps));

    // 4D dot product: 1*4 + 2*(-1) + 3*2 + 4*0.5 = 4 - 2 + 6 + 2 = 10
    EXPECT_FLOAT_EQ(a.Dot(b), 10.0f);
    // Squared magnitude: 1^2 + 2^2 + 3^2 + 4^2 = 30
    EXPECT_FLOAT_EQ(a.SquaredMagnitude(), 30.0f);
    EXPECT_NEAR(a.Magnitude(), std::sqrt(30.0f), kTightEps);
}

TEST(QuaternionTest, NormalizationAndZeroSafety) {
    Quaternion q(1.0f, -2.0f, 3.0f, -4.0f);
    float invLen = 1.0f / std::sqrt(30.0f);
    Quaternion expected(1.0f * invLen, -2.0f * invLen, 3.0f * invLen, -4.0f * invLen);

    // Verifies that rsqrt + 1 Newton-Raphson step matches IEEE-754 std::sqrt precision
    Quaternion normUnchecked = q.NormalizedUnchecked();
    Quaternion normSafe      = q.NormalizedSafe();

    EXPECT_TRUE(normUnchecked.IsNearlyEqual(expected, kTightEps));
    EXPECT_TRUE(normSafe.IsNearlyEqual(expected, kTightEps));
    EXPECT_NEAR(normUnchecked.Magnitude(), 1.0f, kTightEps);

    // Verifies branchless fallback to Identity [0, 0, 0, 1] on zero/sub-epsilon quaternions
    Quaternion zeroQuat(0.0f, 0.0f, 0.0f, 0.0f);
    EXPECT_EQ(zeroQuat.NormalizedSafe(), Quaternion::Identity());
}

TEST(QuaternionTest, HamiltonFundamentalIdentities) {
    // Pure imaginary basis quaternions: i = (1,0,0,0), j = (0,1,0,0), k = (0,0,1,0)
    const Quaternion i(1.0f, 0.0f, 0.0f, 0.0f);
    const Quaternion j(0.0f, 1.0f, 0.0f, 0.0f);
    const Quaternion k(0.0f, 0.0f, 1.0f, 0.0f);
    const Quaternion minusOne(0.0f, 0.0f, 0.0f, -1.0f);

    // i^2 = j^2 = k^2 = ijk = -1
    EXPECT_TRUE((i * i).IsNearlyEqual(minusOne, kTightEps));
    EXPECT_TRUE((j * j).IsNearlyEqual(minusOne, kTightEps));
    EXPECT_TRUE((k * k).IsNearlyEqual(minusOne, kTightEps));
    EXPECT_TRUE((i * j * k).IsNearlyEqual(minusOne, kTightEps));

    // Non-commutative cyclic products: ij = k, ji = -k, jk = i, kj = -i, ki = j, ik = -j
    EXPECT_TRUE((i * j).IsNearlyEqual(k, kTightEps));
    EXPECT_TRUE((j * i).IsNearlyEqual(-k, kTightEps));
    EXPECT_TRUE((j * k).IsNearlyEqual(i, kTightEps));
    EXPECT_TRUE((k * j).IsNearlyEqual(-i, kTightEps));
    EXPECT_TRUE((k * i).IsNearlyEqual(j, kTightEps));
    EXPECT_TRUE((i * k).IsNearlyEqual(-j, kTightEps));
}

TEST(QuaternionTest, ConjugateAndMultiplicativeInverse) {
    // Non-unit quaternion to test general inverse q * q^-1 = Identity
    Quaternion q(2.0f, -3.0f, 1.0f, 4.0f);
    EXPECT_EQ(q.Conjugated(), Quaternion(-2.0f, 3.0f, -1.0f, 4.0f));

    Quaternion invUnchecked = q.InversedUnchecked();
    Quaternion invSafe      = q.InversedSafe();

    EXPECT_TRUE((q * invUnchecked).IsNearlyEqual(Quaternion::Identity(), kTightEps));
    EXPECT_TRUE((invUnchecked * q).IsNearlyEqual(Quaternion::Identity(), kTightEps));
    EXPECT_TRUE(invSafe.IsNearlyEqual(invUnchecked, kTightEps));

    // Degenerate zero quaternion inverse must safely return Identity
    Quaternion zeroQuat(0.0f, 0.0f, 0.0f, 0.0f);
    EXPECT_EQ(zeroQuat.InversedSafe(), Quaternion::Identity());
}

TEST(QuaternionTest, PrincipalAxisVectorRotationsAndNegativeAngles) {
    const Vec3f unitX(1.0f, 0.0f, 0.0f);
    const Vec3f unitY(0.0f, 1.0f, 0.0f);
    const Vec3f unitZ(0.0f, 0.0f, 1.0f);

    // +90 deg around Z rotates +X to +Y
    Quaternion rotZPos90 = Quaternion::FromAngleZ(kHalfPi);
    Vec3f resZPos = rotZPos90 * unitX;
    EXPECT_NEAR(resZPos.x, 0.0f, kEps);
    EXPECT_NEAR(resZPos.y, 1.0f, kEps);
    EXPECT_NEAR(resZPos.z, 0.0f, kEps);

    // -90 deg around Z rotates +X to -Y (verifies odd parity of sin(-theta/2))
    Quaternion rotZNeg90 = Quaternion::FromAngleZ(-kHalfPi);
    Vec3f resZNeg = rotZNeg90 * unitX;
    EXPECT_NEAR(resZNeg.x, 0.0f, kEps);
    EXPECT_NEAR(resZNeg.y, -1.0f, kEps);
    EXPECT_NEAR(resZNeg.z, 0.0f, kEps);

    // +90 deg around X rotates +Y to +Z
    Quaternion rotXPos90 = Quaternion::FromAngleX(kHalfPi);
    Vec3f resXPos = rotXPos90 * unitY;
    EXPECT_NEAR(resXPos.x, 0.0f, kEps);
    EXPECT_NEAR(resXPos.y, 0.0f, kEps);
    EXPECT_NEAR(resXPos.z, 1.0f, kEps);

    // +90 deg around Y rotates +Z to +X
    Quaternion rotYPos90 = Quaternion::FromAngleY(kHalfPi);
    Vec3f resYPos = rotYPos90 * unitZ;
    EXPECT_NEAR(resYPos.x, 1.0f, kEps);
    EXPECT_NEAR(resYPos.y, 0.0f, kEps);
    EXPECT_NEAR(resYPos.z, 0.0f, kEps);
}

TEST(QuaternionTest, DoubleCoverAndRotationComposition) {
    Quaternion q1 = Quaternion::FromAxisAngle(Vec3f(1.0f, 2.0f, 3.0f).Normalized(), 0.85f);
    Quaternion q2 = Quaternion::FromAxisAngle(Vec3f(-2.0f, 1.0f, 0.5f).Normalized(), -1.25f);
    Vec3f v(3.5f, -1.2f, 4.8f);

    // Double cover property: q and -q are algebraically distinct but represent the exact same 3D rotation
    EXPECT_FALSE(q1.IsNearlyEqual(-q1));
    EXPECT_TRUE(q1.IsSameRotation(-q1));
    Vec3f rotPos = q1 * v;
    Vec3f rotNeg = (-q1) * v;
    EXPECT_NEAR(rotPos.x, rotNeg.x, kEps);
    EXPECT_NEAR(rotPos.y, rotNeg.y, kEps);
    EXPECT_NEAR(rotPos.z, rotNeg.z, kEps);

    // Composition homomorphism: (q2 * q1) * v == q2 * (q1 * v)
    Vec3f composedVec   = (q2 * q1) * v;
    Vec3f sequentialVec = q2 * (q1 * v);
    EXPECT_NEAR(composedVec.x, sequentialVec.x, kEps);
    EXPECT_NEAR(composedVec.y, sequentialVec.y, kEps);
    EXPECT_NEAR(composedVec.z, sequentialVec.z, kEps);
}

TEST(QuaternionTest, ToMat4x4fMatchesMatrixRotationsAndVectorTransform) {
    const float angleX = 0.42f;
    const float angleY = -0.78f;
    const float angleZ = 1.15f;

    // Compares ToMat4x4f against Mat4x4f::CreateRotationX/Y/Z
    Mat4x4f qMatX = Quaternion::FromAngleX(angleX).ToMat4x4f();
    Mat4x4f mMatX = Mat4x4f::CreateRotationX(angleX);
    for (size_t col = 0; col < 4; ++col) {
        EXPECT_TRUE(qMatX[col].IsNearlyEqual(mMatX[col], kEps));
    }

    Mat4x4f qMatY = Quaternion::FromAngleY(angleY).ToMat4x4f();
    Mat4x4f mMatY = Mat4x4f::CreateRotationY(angleY);
    for (size_t col = 0; col < 4; ++col) {
        EXPECT_TRUE(qMatY[col].IsNearlyEqual(mMatY[col], kEps));
    }

    Mat4x4f qMatZ = Quaternion::FromAngleZ(angleZ).ToMat4x4f();
    Mat4x4f mMatZ = Mat4x4f::CreateRotationZ(angleZ);
    for (size_t col = 0; col < 4; ++col) {
        EXPECT_TRUE(qMatZ[col].IsNearlyEqual(mMatZ[col], kEps));
    }

    // Verifies Euler Rz * Ry * Rx equivalence between Quaternion::FromEulerAngles and Mat4x4f::CreateRotationXYZ
    Vec3f euler(angleX, angleY, angleZ);
    Quaternion qEuler = Quaternion::FromEulerAngles(euler);
    Quaternion qChain = Quaternion::FromAngleZ(angleZ) * Quaternion::FromAngleY(angleY) * Quaternion::FromAngleX(angleX);
    EXPECT_TRUE(qEuler.IsNearlyEqual(qChain, kEps));

    Mat4x4f qEulerMat = qEuler.ToMat4x4f();
    Mat4x4f mEulerMat = Mat4x4f::CreateRotationXYZ(euler);
    for (size_t col = 0; col < 4; ++col) {
        EXPECT_TRUE(qEulerMat[col].IsNearlyEqual(mEulerMat[col], kEps));
    }

    // Verifies that rotating a 3D vector via Quaternion (q * v) matches Matrix TransformDirection(v)
    Vec3f testDir(2.0f, -5.0f, 1.5f);
    Vec3f rotatedByQuat = qEuler * testDir;
    Vec4f rotatedByMat  = qEulerMat.TransformDirection(testDir);
    EXPECT_NEAR(rotatedByQuat.x, rotatedByMat.x, kEps);
    EXPECT_NEAR(rotatedByQuat.y, rotatedByMat.y, kEps);
    EXPECT_NEAR(rotatedByQuat.z, rotatedByMat.z, kEps);
}

TEST(QuaternionTest, ShepperdFromMat4x4fAllFourBranches) {
    // Tests all 4 branches of Shepperd's algorithm:
    // 1. trace > 0 (small angle)
    // 2. m00 largest (180 deg around X)
    // 3. m11 largest (180 deg around Y)
    // 4. m22 largest (180 deg around Z)
    const Quaternion testCases[] = {
        Quaternion::FromAxisAngle(Vec3f(1.0f, 1.0f, 1.0f).Normalized(), 0.5f),
        Quaternion::FromAngleX(kPi),
        Quaternion::FromAngleY(kPi),
        Quaternion::FromAngleZ(kPi),
        Quaternion::FromAxisAngle(Vec3f(0.95f, -0.2f, 0.1f).Normalized(), 2.95f),
        Quaternion::FromAxisAngle(Vec3f(0.1f, 0.95f, -0.2f).Normalized(), 3.05f),
        Quaternion::FromAxisAngle(Vec3f(-0.2f, 0.1f, 0.95f).Normalized(), 3.10f)
    };

    for (const Quaternion& original : testCases) {
        Mat4x4f rotMat         = original.ToMat4x4f();
        Quaternion roundTripped = Quaternion::FromMat4x4f(rotMat);
        EXPECT_TRUE(original.IsSameRotation(roundTripped, kEps));
    }
}

TEST(QuaternionTest, FromToRotationGeneralParallelAndAntipodal) {
    // General case: arbitrary normalized 3D vectors
    Vec3f from = Vec3f(1.0f, 2.0f, -1.0f).Normalized();
    Vec3f to   = Vec3f(-2.0f, 0.5f, 3.0f).Normalized();
    Quaternion qGen = Quaternion::FromToRotation(from, to);
    Vec3f rotatedGen = qGen * from;
    EXPECT_NEAR(rotatedGen.x, to.x, kEps);
    EXPECT_NEAR(rotatedGen.y, to.y, kEps);
    EXPECT_NEAR(rotatedGen.z, to.z, kEps);

    // Parallel case (0 deg): must return Identity
    Quaternion qParallel = Quaternion::FromToRotation(from, from);
    EXPECT_TRUE(qParallel.IsNearlyEqual(Quaternion::Identity(), kEps));

    // Antipodal case (180 deg) along X and non-X axes
    Vec3f antiA(1.0f, 0.0f, 0.0f);
    Quaternion qAntiX = Quaternion::FromToRotation(antiA, -antiA);
    Vec3f resAntiX = qAntiX * antiA;
    EXPECT_NEAR(resAntiX.x, -antiA.x, kEps);
    EXPECT_NEAR(resAntiX.y, -antiA.y, kEps);
    EXPECT_NEAR(resAntiX.z, -antiA.z, kEps);

    Quaternion qAntiGen = Quaternion::FromToRotation(from, -from);
    Vec3f resAntiGen = qAntiGen * from;
    EXPECT_NEAR(resAntiGen.x, -from.x, kEps);
    EXPECT_NEAR(resAntiGen.y, -from.y, kEps);
    EXPECT_NEAR(resAntiGen.z, -from.z, kEps);
}

TEST(QuaternionTest, LookRotationAndAxisAngleExtraction) {
    Vec3f forward = Vec3f(1.0f, -0.5f, -2.0f).Normalized();
    Vec3f up(0.0f, 1.0f, 0.0f);

    Quaternion lookQuat = Quaternion::LookRotation(forward, up);

    // In right-handed camera convention, local -Z (0, 0, -1) rotated by lookQuat must align with 'forward'
    Vec3f worldForward = lookQuat * Vec3f(0.0f, 0.0f, -1.0f);
    EXPECT_NEAR(worldForward.x, forward.x, kEps);
    EXPECT_NEAR(worldForward.y, forward.y, kEps);
    EXPECT_NEAR(worldForward.z, forward.z, kEps);

    // Axis-Angle round-trip extraction
    Vec3f expectedAxis = Vec3f(-1.0f, 2.0f, 0.5f).Normalized();
    float expectedAngle = 1.35f;
    Quaternion qAxisAngle = Quaternion::FromAxisAngle(expectedAxis, expectedAngle);

    Vec3f extractedAxis;
    float extractedAngle = 0.0f;
    qAxisAngle.ToAxisAngle(extractedAxis, extractedAngle);

    EXPECT_NEAR(extractedAngle, expectedAngle, kEps);
    EXPECT_NEAR(extractedAxis.x, expectedAxis.x, kEps);
    EXPECT_NEAR(extractedAxis.y, expectedAxis.y, kEps);
    EXPECT_NEAR(extractedAxis.z, expectedAxis.z, kEps);

    // Relative angular distance between two orientations
    Quaternion qA = Quaternion::Identity();
    Quaternion qB = Quaternion::FromAngleY(kHalfPi);
    EXPECT_NEAR(Quaternion::Angle(qA, qB), kHalfPi, kEps);
    EXPECT_NEAR(Quaternion::Angle(qA, -qB), kHalfPi, kEps); // Shortest path invariance
}

TEST(QuaternionTest, SlerpAndNlerpShortestPathAndConstantVelocity) {
    Quaternion qStart = Quaternion::FromAngleZ(0.0f);
    Quaternion qEnd   = Quaternion::FromAngleZ(kHalfPi);

    // Boundary checks at t = 0 and t = 1
    EXPECT_TRUE(Quaternion::Slerp(qStart, qEnd, 0.0f).IsSameRotation(qStart, kEps));
    EXPECT_TRUE(Quaternion::Slerp(qStart, qEnd, 1.0f).IsSameRotation(qEnd, kEps));
    EXPECT_TRUE(Quaternion::Nlerp(qStart, qEnd, 0.0f).IsSameRotation(qStart, kEps));
    EXPECT_TRUE(Quaternion::Nlerp(qStart, qEnd, 1.0f).IsSameRotation(qEnd, kEps));

    // Constant angular velocity of Slerp: at t = 0.25, 0.5, 0.75 angle must be exactly t * (pi/2)
    for (float t : {0.25f, 0.5f, 0.75f}) {
        Quaternion qSlerp = Quaternion::Slerp(qStart, qEnd, t);
        Quaternion qExact = Quaternion::FromAngleZ(t * kHalfPi);
        EXPECT_TRUE(qSlerp.IsNearlyEqual(qExact, kEps));
        EXPECT_NEAR(qSlerp.Magnitude(), 1.0f, kEps);
    }

    // Shortest-path hemisphere selection: Slerp(qStart, -qEnd, 0.5) must take the 45-deg short arc, not the 135-deg long arc
    Quaternion qMidPos = Quaternion::Slerp(qStart, qEnd, 0.5f);
    Quaternion qMidNeg = Quaternion::Slerp(qStart, -qEnd, 0.5f);
    EXPECT_TRUE(qMidPos.IsSameRotation(qMidNeg, kEps));

    Quaternion nMidPos = Quaternion::Nlerp(qStart, qEnd, 0.5f);
    Quaternion nMidNeg = Quaternion::Nlerp(qStart, -qEnd, 0.5f);
    EXPECT_TRUE(nMidPos.IsSameRotation(nMidNeg, kEps));

    // Collinear fallback branch (absCosTheta > 0.9995f): must not divide by zero or produce NaN
    Quaternion qTiny = Quaternion::FromAngleX(1e-4f);
    Quaternion qCollinear = Quaternion::Slerp(qStart, qTiny, 0.5f);
    EXPECT_NEAR(qCollinear.Magnitude(), 1.0f, kEps);
    EXPECT_TRUE(qCollinear.IsSameRotation(Quaternion::FromAngleX(0.5e-4f), kEps));
}
