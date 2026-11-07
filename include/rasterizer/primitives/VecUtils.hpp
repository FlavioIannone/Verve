#include "rasterizer/primitives/Vec2f.hpp"
#include "rasterizer/primitives/Vec3f.hpp"
#include "rasterizer/primitives/Vec4f.hpp"


// Vec2f
inline Vec3f Vec2f::ToVec3f(float z) const { return {this->x, this->y, z}; }

inline Vec4f Vec2f::ToVec4f_Point(float z) const { return {this->x, this->y, z, 1.0f}; }

inline Vec4f Vec2f::ToVec4f_Direction(float z) const { return {this->x, this->y, z, 0.0f}; }

inline Vec4f Vec2f::ToVec4f(float z, float w) const { return {this->x, this->y, z, w}; }


// Vec3f
inline Vec2f Vec3f::ToVec2f() const { return {this->x, this->y}; }

inline Vec4f Vec3f::ToVec4f_Direction() const{ return {this->x, this->y, this->z, 0.0f}; }

inline Vec4f Vec3f::ToVec4f_Point() const{ return {this->x, this->y, this->z, 1.0f}; }

inline Vec4f Vec3f::ToVec4f(float w) const { return {this->x, this->y, this->z, w}; }

// Vec4f
inline Vec2f Vec4f::ToVec2f() const { return {this->x, this->y}; }

inline Vec3f Vec4f::ToVec3f() const { return {this->x, this->y, this->z}; }