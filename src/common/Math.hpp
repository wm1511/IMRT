// Copyright Wiktor Merta 2023
// Most of the functions based on NVIDIA helper_math.h from CUDA samples repository
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

#ifndef __CUDACC__
#include <cmath>

__inline__ float fminf(const float a, const float b)
{
	return a < b ? a : b;
}

__inline__ float fmaxf(const float a, const float b)
{
	return a > b ? a : b;
}

__inline__ float rsqrtf(const float x)
{
	return 1.0f / sqrtf(x);
}

#endif

// Constants
__device__ __constant__ constexpr float kPi = 3.141593f;
__device__ __constant__ constexpr float k2Pi = 6.283185f;
__device__ __constant__ constexpr float kHalfPi = 1.570796f;
__device__ __constant__ constexpr float kInvPi = 0.318309f;
__device__ __constant__ constexpr float kInv2Pi = 0.159154f;
__device__ __constant__ constexpr float kTMin = 0.001f;

// Constructors
__inline__ __host__ __device__ float3 make_float3(const float s)
{
	return make_float3(s, s, s);
}

__inline__ __host__ __device__ float3 make_float3(const float t[3])
{
	return make_float3(t[0], t[1], t[2]);
}

__inline__ __host__ __device__ float4 make_float4(const float3 v, const float s)
{
	return make_float4(v.x, v.y, v.z, s);
}

__inline__ __host__ __device__ float4 make_float4(const float t[4])
{
	return make_float4(t[0], t[1], t[2], t[3]);
}

// Negation
__inline__ __host__ __device__ float2 operator-(float2& a)
{
	return make_float2(-a.x, -a.y);
}

__inline__ __host__ __device__ float3 operator-(float3& a)
{
	return make_float3(-a.x, -a.y, -a.z);
}

__inline__ __host__ __device__ float3 operator-(const float3& a)
{
	return make_float3(-a.x, -a.y, -a.z);
}

// Addition
__inline__ __host__ __device__ float2 operator+(const float2 a, const float2 b)
{
	return make_float2(a.x + b.x, a.y + b.y);
}
__inline__ __host__ __device__ void operator+=(float2& a, const float2 b)
{
	a.x += b.x;
	a.y += b.y;
}
__inline__ __host__ __device__ float2 operator+(const float2 a, const float b)
{
	return make_float2(a.x + b, a.y + b);
}
__inline__ __host__ __device__ float2 operator+(const float b, const float2 a)
{
	return make_float2(a.x + b, a.y + b);
}
__inline__ __host__ __device__ void operator+=(float2& a, const float b)
{
	a.x += b;
	a.y += b;
}
__inline__ __host__ __device__ float3 operator+(const float3 a, const float3 b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__inline__ __host__ __device__ float4 operator+(const float4 a, const float4 b)
{
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__inline__ __host__ __device__ void operator+=(float3& a, const float3 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}
__inline__ __host__ __device__ void operator+=(float4& a, const float4 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
}
__inline__ __host__ __device__ float3 operator+(const float3 a, const float b)
{
	return make_float3(a.x + b, a.y + b, a.z + b);
}
__inline__ __host__ __device__ float4 operator+(const float4 a, const float b)
{
	return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
__inline__ __host__ __device__ void operator+=(float3& a, const float b)
{
	a.x += b;
	a.y += b;
	a.z += b;
}
__inline__ __host__ __device__ void operator+=(float4& a, const float b)
{
	a.x += b;
	a.y += b;
	a.z += b;
	a.w += b;
}
__inline__ __host__ __device__ float3 operator+(const float b, const float3 a)
{
	return make_float3(a.x + b, a.y + b, a.z + b);
}
__inline__ __host__ __device__ float4 operator+(const float b, const float4 a)
{
	return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}

// Subtraction
__inline__ __host__ __device__ float2 operator-(const float2 a, const float2 b)
{
	return make_float2(a.x - b.x, a.y - b.y);
}
__inline__ __host__ __device__ void operator-=(float2& a, const float2 b)
{
	a.x -= b.x;
	a.y -= b.y;
}
__inline__ __host__ __device__ float2 operator-(const float2 a, const float b)
{
	return make_float2(a.x - b, a.y - b);
}
__inline__ __host__ __device__ float2 operator-(const float b, const float2 a)
{
	return make_float2(b - a.x, b - a.y);
}
__inline__ __host__ __device__ void operator-=(float2& a, const float b)
{
	a.x -= b;
	a.y -= b;
}
__inline__ __host__ __device__ float3 operator-(const float3 a, const float3 b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__inline__ __host__ __device__ float4 operator-(const float4 a, const float4 b)
{
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
__inline__ __host__ __device__ void operator-=(float3& a, const float3 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}
__inline__ __host__ __device__ void operator-=(float4& a, const float4 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
}
__inline__ __host__ __device__ float3 operator-(const float3 a, const float b)
{
	return make_float3(a.x - b, a.y - b, a.z - b);
}
__inline__ __host__ __device__ float4 operator-(const float4 a, const float b)
{
	return make_float4(a.x - b, a.y - b, a.z - b, a.w - b);
}
__inline__ __host__ __device__ void operator-=(float3& a, const float b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
}
__inline__ __host__ __device__ void operator-=(float4& a, const float b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
	a.w -= b;
}
__inline__ __host__ __device__ float3 operator-(const float b, const float3 a)
{
	return make_float3(b - a.x, b - a.y, b - a.z);
}
__inline__ __host__ __device__ float4 operator-(const float b, const float4 a)
{
	return make_float4(a.x - b, a.y - b, a.z - b, a.w - b);
}

// Multiplication
__inline__ __host__ __device__ float2 operator*(const float2 a, const float2 b)
{
	return make_float2(a.x * b.x, a.y * b.y);
}
__inline__ __host__ __device__ void operator*=(float2& a, const float2 b)
{
	a.x *= b.x;
	a.y *= b.y;
}
__inline__ __host__ __device__ float2 operator*(const float2 a, const float b)
{
	return make_float2(a.x * b, a.y * b);
}
__inline__ __host__ __device__ float2 operator*(const float b, const float2 a)
{
	return make_float2(b * a.x, b * a.y);
}
__inline__ __host__ __device__ void operator*=(float2& a, const float b)
{
	a.x *= b;
	a.y *= b;
}
__inline__ __host__ __device__ float3 operator*(const float3 a, const float3 b)
{
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__inline__ __host__ __device__ float4 operator*(const float4 a, const float4 b)
{
	return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
__inline__ __host__ __device__ void operator*=(float3& a, const float3 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
}
__inline__ __host__ __device__ void operator*=(float4& a, const float4 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
}
__inline__ __host__ __device__ float3 operator*(const float3 a, const float b)
{
	return make_float3(a.x * b, a.y * b, a.z * b);
}
__inline__ __host__ __device__ float4 operator*(const float4 a, const float b)
{
	return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}
__inline__ __host__ __device__ void operator*=(float3& a, const float b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
}
__inline__ __host__ __device__ void operator*=(float4& a, const float b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
	a.w *= b;
}
__inline__ __host__ __device__ float3 operator*(const float b, const float3 a)
{
	return make_float3(b * a.x, b * a.y, b * a.z);
}
__inline__ __host__ __device__ float4 operator*(const float b, const float4 a)
{
	return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

// Division
__inline__ __host__ __device__ float2 operator/(const float2 a, const float2 b)
{
	return make_float2(a.x / b.x, a.y / b.y);
}
__inline__ __host__ __device__ void operator/=(float2& a, const float2 b)
{
	a.x /= b.x;
	a.y /= b.y;
}
__inline__ __host__ __device__ float2 operator/(const float2 a, const float b)
{
	return make_float2(a.x / b, a.y / b);
}
__inline__ __host__ __device__ void operator/=(float2& a, const float b)
{
	a.x /= b;
	a.y /= b;
}
__inline__ __host__ __device__ float2 operator/(const float b, const float2 a)
{
	return make_float2(b / a.x, b / a.y);
}
__inline__ __host__ __device__ float3 operator/(const float3 a, const float3 b)
{
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
__inline__ __host__ __device__ float4 operator/(const float4 a, const float4 b)
{
	return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
__inline__ __host__ __device__ void operator/=(float3& a, const float3 b)
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
}
__inline__ __host__ __device__ void operator/=(float4& a, const float4 b)
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	a.w /= b.w;
}
__inline__ __host__ __device__ float3 operator/(const float3 a, const float b)
{
	return make_float3(a.x / b, a.y / b, a.z / b);
}
__inline__ __host__ __device__ float4 operator/(const float4 a, const float b)
{
	return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}
__inline__ __host__ __device__ void operator/=(float3& a, const float b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
}
__inline__ __host__ __device__ void operator/=(float4& a, const float b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
	a.w /= b;
}
__inline__ __host__ __device__ float3 operator/(const float b, const float3 a)
{
	return make_float3(b / a.x, b / a.y, b / a.z);
}
__inline__ __host__ __device__ float4 operator/(const float b, const float4 a)
{
	return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}

// Minimum
__inline__  __host__ __device__ float2 fminf(const float2 a, const float2 b)
{
	return make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
}
__inline__  __host__ __device__ float2 fminf(const float2 a, const float2 b, const float2 c)
{
	return make_float2(fminf(c.x, fminf(a.x, b.x)), fminf(c.y, fminf(a.y, b.y)));
}
__inline__ __host__ __device__ float3 fminf(const float3 a, const float3 b)
{
	return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}
__inline__  __host__ __device__ float3 fminf(const float3 a, const float3 b, const float3 c)
{
	return make_float3(fminf(c.x, fminf(a.x, b.x)), fminf(c.y, fminf(a.y, b.y)), fminf(c.z, fminf(a.z, b.z)));
}
__inline__  __host__ __device__ float4 fminf(const float4 a, const float4 b)
{
	return make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w));
}
__inline__  __host__ __device__ float4 fminf(const float4 a, const float4 b, const float4 c)
{
	return make_float4(fminf(c.x, fminf(a.x, b.x)), fminf(c.y, fminf(a.y, b.y)), fminf(c.z, fminf(a.z, b.z)), fminf(c.w, fminf(a.w, b.w)));
}

// Maximum
__inline__ __host__ __device__ float2 fmaxf(const float2 a, const float2 b)
{
	return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}
__inline__  __host__ __device__ float2 fmaxf(const float2 a, const float2 b, const float2 c)
{
	return make_float2(fmaxf(c.x, fmaxf(a.x, b.x)), fmaxf(c.y, fmaxf(a.y, b.y)));
}
__inline__ __host__ __device__ float3 fmaxf(const float3 a, const float3 b)
{
	return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}
__inline__  __host__ __device__ float3 fmaxf(const float3 a, const float3 b, const float3 c)
{
	return make_float3(fmaxf(c.x, fmaxf(a.x, b.x)), fmaxf(c.y, fmaxf(a.y, b.y)), fmaxf(c.z, fmaxf(a.z, b.z)));
}
__inline__ __host__ __device__ float4 fmaxf(const float4 a, const float4 b)
{
	return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
}
__inline__  __host__ __device__ float4 fmaxf(const float4 a, const float4 b, const float4 c)
{
	return make_float4(fmaxf(c.x, fmaxf(a.x, b.x)), fmaxf(c.y, fmaxf(a.y, b.y)), fmaxf(c.z, fmaxf(a.z, b.z)), fmaxf(c.w, fmaxf(a.w, b.w)));
}

// Maximal component
__inline__ __host__ __device__ float maxcomp(const float2 v)
{
	return fmaxf(v.x, v.y);
}

__inline__ __host__ __device__ float maxcomp(const float3 v)
{
	return fmaxf(fmaxf(v.x, v.y), v.z);
}

__inline__ __host__ __device__ float maxcomp(const float4 v)
{
	return fmaxf(fmaxf(fmaxf(v.x, v.y), v.z), v.w);
}

// Minimal component
__inline__ __host__ __device__ float mincomp(const float2 v)
{
	return fminf(v.x, v.y);
}

__inline__ __host__ __device__ float mincomp(const float3 v)
{
	return fminf(fminf(v.x, v.y), v.z);
}

__inline__ __host__ __device__ float mincomp(const float4 v)
{
	return fminf(fminf(fminf(v.x, v.y), v.z), v.w);
}

// Linear interpolation
__inline__ __device__ __host__ float lerp(const float a, const float b, const float t)
{
	return a + t * (b - a);
}
__inline__ __device__ __host__ float2 lerp(const float2 a, const float2 b, const float t)
{
	return a + t * (b - a);
}
__inline__ __device__ __host__ float3 lerp(const float3 a, const float3 b, const float t)
{
	return a + t * (b - a);
}
__inline__ __device__ __host__ float4 lerp(const float4 a, const float4 b, const float t)
{
	return a + t * (b - a);
}

// Fraction
__inline__ __host__ __device__ float fracf(const float v)
{
	return v - floorf(v);
}
__inline__ __host__ __device__ float2 fracf(const float2 v)
{
	return make_float2(fracf(v.x), fracf(v.y));
}
__inline__ __host__ __device__ float3 fracf(const float3 v)
{
	return make_float3(fracf(v.x), fracf(v.y), fracf(v.z));
}
__inline__ __host__ __device__ float4 fracf(const float4 v)
{
	return make_float4(fracf(v.x), fracf(v.y), fracf(v.z), fracf(v.w));
}

// Clamp
__inline__ __host__ __device__ float clamp(const float f, const float a, const float b)
{
	return fmaxf(a, fminf(f, b));
}
__inline__ __host__ __device__ float3 clamp(const float3 v, const float a, const float b)
{
	return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
__inline__ __host__ __device__ float3 clamp(const float3 v, const float3 a, const float3 b)
{
	return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}

// Dot product
__inline__ __host__ __device__ float dot(const float2 a, const float2 b)
{
	return a.x * b.x + a.y * b.y;
}
__inline__ __host__ __device__ float dot(const float3 a, const float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
__inline__ __host__ __device__ float dot(const float4 a, const float4 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// Length
__inline__ __host__ __device__ float length(const float2 v)
{
	return sqrtf(dot(v, v));
}
__inline__ __host__ __device__ float length(const float3 v)
{
	return sqrtf(dot(v, v));
}
__inline__ __host__ __device__ float length(const float4 v)
{
	return sqrtf(dot(v, v));
}


// Normalization
__inline__ __host__ __device__ float3 normalize(const float3 v)
{
	return v * rsqrtf(dot(v, v));
}

// Floor
__inline__ __host__ __device__ float3 floor(const float3 v)
{
	return make_float3(floorf(v.x), floorf(v.y), floorf(v.z));
}

// Ceil
__inline__ __host__ __device__ float3 ceil(const float3 v)
{
	return make_float3(ceilf(v.x), ceilf(v.y), ceilf(v.z));
}

// Floor
__inline__ __host__ __device__ float3 round(const float3 v)
{
	return make_float3(roundf(v.x), roundf(v.y), roundf(v.z));
}

// Absolute value
__inline__ __host__ __device__ float3 abs(const float3 v)
{
	return make_float3(fabs(v.x), fabs(v.y), fabs(v.z));
}

// Square root
__inline__ __host__ __device__ float3 sqrt(const float3 v)
{
	return make_float3(sqrt(v.x), sqrt(v.y), sqrt(v.z));
}

__inline__ __host__ __device__ float4 sqrt(const float4 v)
{
	return make_float4(sqrt(v.x), sqrt(v.y), sqrt(v.z), sqrt(v.w));
}

// Reflection
__inline__ __host__ __device__ float3 reflect(const float3 i, const float3 n)
{
	return i - 2.0f * n * dot(n, i);
}

// Cross product
__inline__ __host__ __device__ float3 cross(const float3 a, const float3 b)
{
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

// Versor
__inline__ __host__ __device__ float3 versor(const float3 v)
{
	return float3(v / length(v));
}

// Transformations
__inline__ __host__ __device__ void translate_point(float3& v, const float3 t)
{
	v += t;
}

__inline__ __host__ __device__ void scale_point(float3& v, const float3 s)
{
	v *= s;
}

__inline__ __host__ __device__ void rotate_point_x(float3& v, const float rx)
{
	const float sc = sin(rx);
	const float cc = cos(rx);

	v = { v.x, cc * v.y - sc * v.z, sc * v.y + cc * v.z };
}

__inline__ __host__ __device__ void rotate_point_y(float3& v, const float ry)
{
	const float sb = sin(ry);
	const float cb = cos(ry);

	v = { cb * v.x - sb * v.z, v.y, sb * v.x + cb * v.z };
}

__inline__ __host__ __device__ void rotate_point_z(float3& v, const float rz)
{
	const float sa = sin(rz);
	const float ca = cos(rz);

	v = { ca * v.x - sa * v.y, sa * v.x + ca * v.y, v.z };
}

__inline__ __host__ __device__ float3 transform_point(float3 v, const float3 t, const float3 s, const float3 r)
{
	scale_point(v, s);

	if (r.x > 0.0f)
		rotate_point_x(v, r.x);

	if (r.y > 0.0f)
		rotate_point_y(v, r.y);

	if (r.z > 0.0f)
		rotate_point_z(v, r.z);

	translate_point(v, t);

	return v;
}

// Fill matrix for usage with Optix instances
__inline__ __host__ __device__ void fill_matrix(float matrix[12], const float3 t, const float3 s, const float3 r)
{
	const float sa = sin(r.z);
	const float ca = cos(r.z);
	const float sb = sin(r.y);
	const float cb = cos(r.y);
	const float sc = sin(r.x);
	const float cc = cos(r.x);

	matrix[0] = cb * cc * s.x;
	matrix[1] = -cb * sc * s.y;
	matrix[2] = sb * s.z;
	matrix[3] = t.x;
	matrix[4] = (sa * sb * cc + ca * sc) * s.x;
	matrix[5] = (-sa * sb * sc + ca * cc) * s.y;
	matrix[6] = -sa * cb * s.z;
	matrix[7] = t.y;
	matrix[8] = (-ca * sb * cc + sa * sc) * s.x;
	matrix[9] = (ca * sb * sc + sa * cc) * s.y;
	matrix[10] = ca * cb * s.z;
	matrix[11] = t.z;
}

// Random
__inline__ __host__ __device__ uint32_t rotl(const uint32_t x, const int k)
{
	return (x << k) | (x >> (32 - k));
}

// pcg_rxs_m_xs from "PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for Random Number Generation"
__inline__ __host__ __device__ float pcg(uint32_t* random_state)
{
	uint32_t state = *random_state;
	*random_state = *random_state * 747796405u + 2891336453u;
	uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (float)(((word >> 22u) ^ word) >> 8) * (1.0f / (UINT32_C(1) << 24));
}

// xoshiro128+ by David Blackman and Sebastiano Vigna, 2018
__inline__ __host__ __device__ uint32_t xoshiro(uint4* random_state)
{
	const uint32_t result = random_state->x + random_state->w;
	const uint32_t t = random_state->y << 9;

	// xo
	random_state->z ^= random_state->x;
	random_state->w ^= random_state->y;
	random_state->y ^= random_state->z;
	random_state->x ^= random_state->w;

	// shi
	random_state->z ^= t;

	// ro
	random_state->w = rotl(random_state->w, 11);

	return result;
}

__inline__ __host__ __device__ float2 disk_random(uint32_t* random_state)
{
	float2 v;
	do
	{
		v = 2.0f * make_float2(pcg(random_state), pcg(random_state)) - make_float2(1.0f, 1.0f);
	} while (dot(v, v) >= 1.0f);
	return v;
}

__inline__ __host__ __device__ float3 sphere_random(uint32_t* random_state)
{
	float3 v;
	do
	{
		v = make_float3(pcg(random_state), pcg(random_state), pcg(random_state)) - make_float3(1.0f);
	} while (dot(v, v) >= 1.0f);
	return v;
}