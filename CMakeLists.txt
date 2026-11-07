cmake_minimum_required(VERSION 3.22)

project(Verve
    VERSION 1.0.0
    DESCRIPTION "Modern C++20 cross-platform software rasterizer with SIMD vectorization (x86_64 / ARM64) and FMA math."
    LANGUAGES CXX
)

# C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Target Architecture Detection
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86|x86_64|AMD64|i386") 
    set(IS_X86 TRUE)    # Target: x86
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64|ARM") 
    set(IS_ARM TRUE)    # Target: ARM
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Android") 
    # Searches for SDL2 through FetchContent
    include(FetchContent) 
    FetchContent_Declare( SDL2 GIT_REPOSITORY https://github.com/libsdl-org/SDL.git GIT_TAG release-2.30.x )
    FetchContent_MakeAvailable(SDL2) 
else()
    # Searches the SDL2 library through vcpkg 
    find_package(SDL2 REQUIRED)
endif()

# Interface library for compile flags
add_library(rasterizer_compiler_flags INTERFACE)

# 1. x86 / x86-64 Architecture (AVX2 + FMA3)
if(IS_X86)
    if(MSVC)
        target_compile_options(rasterizer_compiler_flags INTERFACE
            /arch:AVX2 /permissive- /Zc:preprocessor
            $<$<CONFIG:Release>:/O2 /Ob2 /fp:fast>
            $<$<CONFIG:Debug>:/W4 /Qvec-report:2>
        )
        add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(rasterizer_compiler_flags INTERFACE
            -mavx2 -mfma
            $<$<CONFIG:Release>:-O3 -finline-functions -ffast-math>
            $<$<CONFIG:Debug>:-Wall -Wextra -Wpedantic>
            $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU>>:-fopt-info-vec-optimized>
            $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:Clang>>:-Rpass=loop-vectorize>
        )
    endif()

# 2. ARM64 Architecture (ARM NEON)
elseif(IS_ARM)
    if(MSVC)
        target_compile_options(rasterizer_compiler_flags INTERFACE
            /permissive- /Zc:preprocessor
            $<$<CONFIG:Release>:/O2 /Ob2 /fp:fast>
            $<$<CONFIG:Debug>:/W4>
        )
        add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(rasterizer_compiler_flags INTERFACE
            $<$<CONFIG:Release>:-O3 -finline-functions -ffast-math>
            $<$<CONFIG:Debug>:-Wall -Wextra -Wpedantic>
            $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU>>:-fopt-info-vec-optimized>
            $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:Clang>>:-Rpass=loop-vectorize>
        )
    endif()
endif()


# Executable
add_executable(${PROJECT_NAME} apps/Main.cpp)

# Headers folders (include + vendor)
target_include_directories(rasterizer_compiler_flags INTERFACE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/vendor
)

# Link SDL2 libraries
target_link_libraries(${PROJECT_NAME} PRIVATE 
    $<IF:$<TARGET_EXISTS:SDL2::SDL2main>,SDL2::SDL2main,>
    $<IF:$<TARGET_EXISTS:SDL2::SDL2-static>,SDL2::SDL2-static,SDL2::SDL2>
    rasterizer_compiler_flags
)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Android")
    # Enable tests (GTest)
    enable_testing()
    # Link test subdirectory
    add_subdirectory(test)
endif()