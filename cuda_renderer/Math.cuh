#pragma once
#include "cuda_runtime.h"

#ifndef __CUDACC__
#include <cmath>
#endif

// Constructors
inline __host__ __device__ float3 make_float3(float s)
{
    return make_float3(s, s, s);
}

inline __host__ __device__ float3 make_float3(const float t[3])
{
	return make_float3(t[0], t[1], t[2]);
}

inline __host__ __device__ float4 make_float4(float3 v, float s)
{
    return make_float4(v.x, v.y, v.z, s);
}

// Negation
inline __host__ __device__ float3 operator-(float3 &a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

// Addition
inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(float3 &a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ float3 operator+(float3 a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(float3 &a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}
inline __host__ __device__ float3 operator+(float b, float3 a)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}

// Subtraction
inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(float3 &a, float3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __host__ __device__ float3 operator-(float3 a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ float3 operator-(float b, float3 a)
{
    return make_float3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(float3 &a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

// Multiplication
inline __host__ __device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(float3 &a, float3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline __host__ __device__ float3 operator*(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ float3 operator*(float b, float3 a)
{
    return make_float3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(float3 &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

// Division
inline __host__ __device__ float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ void operator/=(float3 &a, float3 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}
inline __host__ __device__ float3 operator/(float3 a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ void operator/=(float3 &a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
}
inline __host__ __device__ float3 operator/(float b, float3 a)
{
    return make_float3(b / a.x, b / a.y, b / a.z);
}

// Minimum
inline __host__ __device__ float3 min(float3 a, float3 b)
{
    return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}

// Maximum
inline __host__ __device__ float3 max(float3 a, float3 b)
{
    return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}

// Clamp
inline __device__ __host__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}
inline __device__ __host__ float3 clamp(float3 v, float a, float b)
{
    return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ float3 clamp(float3 v, float3 a, float3 b)
{
    return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}

// Dot product
inline __host__ __device__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Length
inline __host__ __device__ float length(float3 v)
{
    return sqrtf(dot(v, v));
}

// Normalization
inline __host__ __device__ float3 normalize(float3 v)
{
    return v * (1.0f / sqrtf(dot(v, v)));
}

// Floor
inline __host__ __device__ float3 floor(float3 v)
{
    return make_float3(floorf(v.x), floorf(v.y), floorf(v.z));
}

// Absolute value
inline __host__ __device__ float3 abs(float3 v)
{
    return make_float3(fabs(v.x), fabs(v.y), fabs(v.z));
}

inline __host__ __device__ float3 sqrt(float3 v)
{
	return make_float3(sqrt(v.x), sqrt(v.y), sqrt(v.z));
}

// Reflection
inline __host__ __device__ float3 reflect(float3 i, float3 n)
{
    return i - 2.0f * n * dot(n,i);
}

// Cross product
inline __host__ __device__ float3 cross(float3 a, float3 b)
{
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

// Versor
inline __host__ __device__ float3 versor(float3 v)
{
	return float3(v / length(v));
}