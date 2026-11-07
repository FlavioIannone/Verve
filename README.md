# Verve

**Verve** is a high-performance, cross-platform software rasterization library written in modern C++20. Designed to explore extreme low-level optimizations and real-time graphics pipelines, it targets maximum CPU throughput across both **x86_64** and **ARM64** architectures.

---

## Key Architectural Features

- **Projective Space Algebra ($\mathbb{R}^4$):** Vector math and homogeneous coordinates optimized for 3D transformations.
- **Column-Major Matrices:** Linear combinations of column vectors calculated via Vector Broadcast and **Fused Multiply-Add (FMA)** hardware instructions.
- **SIMD Acceleration:** Universal cross-platform vectorization utilizing **SIMDe (SIMD Everywhere)** mapped natively to **AVX2 / FMA3** on x86_64 and **ARM NEON** on ARM64.
- **Hardware-Aligned Memory:** Strict 32-byte alignment across primitive math types to prevent Cache Line Splits and enable fast SIMD loads.
- **Cross-Platform Build System:** Fully integrated CMake build pipeline supporting **MSVC (Windows x64/ARM64)** and **Clang (Linux/Android Termux)** with interactive SDL2 rendering.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

_Note: This is an initial development placeholder. Complete technical specifications, benchmarks, and API documentation will be published upon initial library release._
