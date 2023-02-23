#pragma once
#include <cuda_runtime.h>

#ifndef __CUDACC__
#include <cmath>

inline float fminf(float a, float b)
{
	return a < b ? a : b;
}

inline float fmaxf(float a, float b)
{
	return a > b ? a : b;
}

inline float rsqrtf(float x)
{
	return 1.0f / sqrtf(x);
}

#endif

// Constants
__device__ __constant__ constexpr float kPi = 3.141593f;
__device__ __constant__ constexpr float kTwoPi = 6.283185f;
__device__ __constant__ constexpr float kHalfPi =  1.570796f;
__device__ __constant__ constexpr float kInvPi = 0.318309f;
__device__ __constant__ constexpr float kInv2Pi = 0.159154f;

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

inline __host__ __device__ float4 make_float4(const float t[4])
{
	return make_float4(t[0], t[1], t[2], t[3]);
}

// Negation
inline __host__ __device__ float3 operator-(float3 &a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

inline __host__ __device__ float3 operator-(const float3 &a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

// Addition
inline __host__ __device__ float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(float2 &a, float2 b)
{
    a.x += b.x;
    a.y += b.y;
}
inline __host__ __device__ float2 operator+(float2 a, float b)
{
    return make_float2(a.x + b, a.y + b);
}
inline __host__ __device__ float2 operator+(float b, float2 a)
{
    return make_float2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(float2 &a, float b)
{
    a.x += b;
    a.y += b;
}
inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ void operator+=(float3 &a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline __host__ __device__ float3 operator+(float3 a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ float4 operator+(float4 a, float b)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ void operator+=(float3 &a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}
inline __host__ __device__ void operator+=(float4 &a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}
inline __host__ __device__ float3 operator+(float b, float3 a)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ float4 operator+(float b, float4 a)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}

// Subtraction
inline __host__ __device__ float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(float2 &a, float2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline __host__ __device__ float2 operator-(float2 a, float b)
{
    return make_float2(a.x - b, a.y - b);
}
inline __host__ __device__ float2 operator-(float b, float2 a)
{
    return make_float2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(float2 &a, float b)
{
    a.x -= b;
    a.y -= b;
}
inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline __host__ __device__ void operator-=(float3 &a, float3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __host__ __device__ void operator-=(float4 &a, float4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline __host__ __device__ float3 operator-(float3 a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ float4 operator-(float4 a, float b)
{
    return make_float4(a.x - b, a.y - b, a.z - b, a.w - b);
}
inline __host__ __device__ void operator-=(float3 &a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}
inline __host__ __device__ void operator-=(float4 &a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}
inline __host__ __device__ float3 operator-(float b, float3 a)
{
    return make_float3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ float4 operator-(float b, float4 a)
{
	return make_float4(a.x - b, a.y - b, a.z - b, a.w - b);
}

// Multiplication
inline __host__ __device__ float2 operator*(float2 a, float2 b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(float2 &a, float2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
inline __host__ __device__ float2 operator*(float2 a, float b)
{
    return make_float2(a.x * b, a.y * b);
}
inline __host__ __device__ float2 operator*(float b, float2 a)
{
    return make_float2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(float2 &a, float b)
{
    a.x *= b;
    a.y *= b;
}
inline __host__ __device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __host__ __device__ void operator*=(float3 &a, float3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline __host__ __device__ void operator*=(float4 &a, float4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline __host__ __device__ float3 operator*(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ float4 operator*(float4 a, float b)
{
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}
inline __host__ __device__ void operator*=(float3 &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}
inline __host__ __device__ void operator*=(float4 &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}
inline __host__ __device__ float3 operator*(float b, float3 a)
{
    return make_float3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ float4 operator*(float b, float4 a)
{
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

// Division
inline __host__ __device__ float2 operator/(float2 a, float2 b)
{
    return make_float2(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ void operator/=(float2 &a, float2 b)
{
    a.x /= b.x;
    a.y /= b.y;
}
inline __host__ __device__ float2 operator/(float2 a, float b)
{
    return make_float2(a.x / b, a.y / b);
}
inline __host__ __device__ void operator/=(float2 &a, float b)
{
    a.x /= b;
    a.y /= b;
}
inline __host__ __device__ float2 operator/(float b, float2 a)
{
    return make_float2(b / a.x, b / a.y);
}
inline __host__ __device__ float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ float4 operator/(float4 a, float4 b)
{
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
inline __host__ __device__ void operator/=(float3 &a, float3 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}
inline __host__ __device__ void operator/=(float4 &a, float4 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
}
inline __host__ __device__ float3 operator/(float3 a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ float4 operator/(float4 a, float b)
{
    return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}
inline __host__ __device__ void operator/=(float3 &a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
}
inline __host__ __device__ void operator/=(float4 &a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}
inline __host__ __device__ float3 operator/(float b, float3 a)
{
    return make_float3(b / a.x, b / a.y, b / a.z);
}
inline __host__ __device__ float4 operator/(float b, float4 a)
{
    return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}

// Minimum
inline  __host__ __device__ float2 fminf(float2 a, float2 b)
{
    return make_float2(fminf(a.x,b.x), fminf(a.y,b.y));
}
inline  __host__ __device__ float2 fminf(float2 a, float2 b, float2 c)
{
    return make_float2(fminf(c.x, fminf(a.x,b.x)), fminf(c.y, fminf(a.y,b.y)));
}
inline __host__ __device__ float3 fminf(float3 a, float3 b)
{
    return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}
inline  __host__ __device__ float3 fminf(float3 a, float3 b, float3 c)
{
    return make_float3(fminf(c.x, fminf(a.x,b.x)), fminf(c.y, fminf(a.y,b.y)), fminf(c.z, fminf(a.z,b.z)));
}
inline  __host__ __device__ float4 fminf(float4 a, float4 b)
{
    return make_float4(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z), fminf(a.w,b.w));
}
inline  __host__ __device__ float4 fminf(float4 a, float4 b, float4 c)
{
    return make_float4(fminf(c.x, fminf(a.x,b.x)), fminf(c.y, fminf(a.y,b.y)), fminf(c.z, fminf(a.z,b.z)), fminf(c.w, fminf(a.w,b.w)));
}

// Maximum
inline __host__ __device__ float2 fmaxf(float2 a, float2 b)
{
    return make_float2(fmaxf(a.x,b.x), fmaxf(a.y,b.y));
}
inline  __host__ __device__ float2 fmaxf(float2 a, float2 b, float2 c)
{
    return make_float2(fmaxf(c.x, fmaxf(a.x,b.x)), fmaxf(c.y, fmaxf(a.y,b.y)));
}
inline __host__ __device__ float3 fmaxf(float3 a, float3 b)
{
    return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}
inline  __host__ __device__ float3 fmaxf(float3 a, float3 b, float3 c)
{
    return make_float3(fmaxf(c.x, fmaxf(a.x,b.x)), fmaxf(c.y, fmaxf(a.y,b.y)), fmaxf(c.z, fmaxf(a.z,b.z)));
}
inline __host__ __device__ float4 fmaxf(float4 a, float4 b)
{
    return make_float4(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z), fmaxf(a.w,b.w));
}
inline  __host__ __device__ float4 fmaxf(float4 a, float4 b, float4 c)
{
    return make_float4(fmaxf(c.x, fmaxf(a.x,b.x)), fmaxf(c.y, fmaxf(a.y,b.y)), fmaxf(c.z, fmaxf(a.z,b.z)), fmaxf(c.w, fmaxf(a.w,b.w)));
}

// Linear interpolation
inline __device__ __host__ float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float2 lerp(float2 a, float2 b, float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float3 lerp(float3 a, float3 b, float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float4 lerp(float4 a, float4 b, float t)
{
    return a + t*(b-a);
}

// Clamp
inline __host__ __device__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}
inline __host__ __device__ float3 clamp(float3 v, float a, float b)
{
    return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __host__ __device__ float3 clamp(float3 v, float3 a, float3 b)
{
    return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}

// Dot product
inline __host__ __device__ float dot(float2 a, float2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ float dot(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// Length
inline __host__ __device__ float length(float3 v)
{
    return sqrtf(dot(v, v));
}

// Normalization
inline __host__ __device__ float3 normalize(float3 v)
{
    return v * rsqrtf(dot(v, v));
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

// Square root
inline __host__ __device__ float3 sqrt(float3 v)
{
	return make_float3(sqrt(v.x), sqrt(v.y), sqrt(v.z));
}

inline __host__ __device__ float4 sqrt(float4 v)
{
	return make_float4(sqrt(v.x), sqrt(v.y), sqrt(v.z), sqrt(v.w));
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

// Transformations
inline __host__ __device__ float3 point_transform(float3 v, float3 t, float3 s, float3 r)
{
	const float sa = sin(r.x); 
	const float ca = cos(r.x); 
	const float sb = sin(r.y); 
	const float cb = cos(r.y); 
	const float sc = sin(r.z); 
	const float cc = cos(r.z);

	v += t;
	v *= s;
	v.x = ca * cb * v.x - sa * cc * v.y - ca * sb * sc * v.y + sa * sc * v.z - ca * sb * cc * v.z;
	v.y = sa * cb * v.x + ca * cc * v.y - sa * sb * sc * v.y - ca * sc * v.z - sa * sb * cc * v.z;
	v.z = sb * v.x + cb * sc * v.y + cb * cc * v.z;

    return v;
}

// Random
inline __host__ __device__ uint32_t rotl(const uint32_t x, int k)
{
	return (x << k) | (x >> (32 - k));
}

// pcg_rxs_m_xs
inline __host__ __device__ float pcg(uint32_t* random_state)
{
    uint32_t state = *random_state;
    *random_state = *random_state * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (float)(((word >> 22u) ^ word) >> 8) * (1.0f / (UINT32_C(1) << 24));
}

// xoshiro128+
inline __host__ __device__ uint32_t xoshiro(uint4* random_state)
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

inline __host__ __device__ float3 disk_random(uint32_t* random_state)
{
	float3 v;
	do
	{
		v = 2.0f * make_float3(pcg(random_state), pcg(random_state), 0.0f) - make_float3(1.0f, 1.0f, 0.0f);
	} while (dot(v, v) >= 1.0f);
	return v;
}

inline __host__ __device__ float3 sphere_random(uint32_t* random_state)
{
	float3 v;
	do
	{
		v = make_float3(pcg(random_state), pcg(random_state), pcg(random_state)) - make_float3(1.0f);
	} while (dot(v, v) >= 1.0f);
	return v;
}