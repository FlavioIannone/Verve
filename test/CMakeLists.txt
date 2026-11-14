# Find GoogleTest library
find_package(GTest REQUIRED)

# Add the unit test executable
add_executable(unit_tests 
    # ./TestMain.cpp
    # ${CMAKE_SOURCE_DIR}/src/primitives/tests/AlignTest.cpp
    # ${CMAKE_SOURCE_DIR}/src/primitives/tests/SimdOperationsCorrectnessTest.cpp
    # ${CMAKE_SOURCE_DIR}/src/primitives/tests/Vec2fTest.cpp
    # ${CMAKE_SOURCE_DIR}/src/primitives/tests/Vec3fTest.cpp
    # ${CMAKE_SOURCE_DIR}/src/primitives/tests/Vec4fTest.cpp
    # ${CMAKE_SOURCE_DIR}/src/primitives/tests/Mat4x4fTest.cpp
    ${CMAKE_SOURCE_DIR}/src/primitives/tests/QuaternionTest.cpp
)

target_include_directories(unit_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Include libaries
target_link_libraries(unit_tests PRIVATE 
    rasterizer_compiler_flags
    $<IF:$<TARGET_EXISTS:GTest::gtest_main>,GTest::gtest_main,>
    $<IF:$<TARGET_EXISTS:GTest::gtest>,GTest::gtest,>
)

# Include GoogleTest helper module for CMake
include(GoogleTest)

# Register tests
if(NOT CMAKE_CROSSCOMPILING)
    gtest_discover_tests(unit_tests)
endif()