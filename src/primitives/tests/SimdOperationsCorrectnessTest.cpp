#include <gtest/gtest.h>
#include <iostream>
#include <rasterizer/math_core/Primitives.hpp>
#define ITERATIONS 10000000


TEST(SIMD_TESTS, OPERATIONS_CORRECTNESS) {
    Vec4f scalarAcc = Vec4f();
    Vec4f v1 = Vec4f(0.25f, 0.5f, 0.125f, 0.0625f); // Powers of 2, to avoid altering the floating-point result due to accumulation
    Vec4f v2 = Vec4f(1);
    
    // SIMD accumulators. Use 4 accumulators, Loop Unrolling, num_of_accumulators = nums_of_cycles_per_instruction (4 cycles for fmadd) / Throughput (1 cycle)
    Vec4f simdAcc = Vec4f();
    Vec4f acc0 = Vec4f();
    Vec4f acc1 = Vec4f();
    Vec4f acc2 = Vec4f();
    Vec4f acc3 = Vec4f();
    
    
    for (auto i = 0; i < ITERATIONS; i++) {
        scalarAcc.x += v1.x * v2.x;
        scalarAcc.y += v1.y * v2.y;
        scalarAcc.z += v1.z * v2.z;
        scalarAcc.w += v1.w * v2.w;
    }
    
    // SIMD
    // Because of the 4 accumulators, the loop now needs to do 4 times less iterations to get to the same result
    for (auto i = 0; i < ITERATIONS / 4 ; i++) {
        acc0.fmadd(v1, v2);
        acc1.fmadd(v1, v2);
        acc2.fmadd(v1, v2);
        acc3.fmadd(v1, v2);
    }
    // Sum the accumulators
    simdAcc = acc0 + acc1 + acc2 + acc3;

    // SIMD
    std::cout << "[SIMD]: Vector acc: " << simdAcc.x << ", " << simdAcc.y << ", " << simdAcc.z << ", " << simdAcc.w << std::endl;

    // Scalar
    std::cout << "[SCALAR]: Vector acc: " << scalarAcc.x << ", " << scalarAcc.y << ", " << scalarAcc.z << ", " << scalarAcc.w << std::endl;

    // The two outputs must be equal
    EXPECT_EQ(simdAcc.x, scalarAcc.x);
    EXPECT_EQ(simdAcc.y, scalarAcc.y);
    EXPECT_EQ(simdAcc.z, scalarAcc.z);
    EXPECT_EQ(simdAcc.w, scalarAcc.w);
}
