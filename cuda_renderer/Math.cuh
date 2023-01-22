#pragma once
#include <cuda_runtime.h>

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

inline __host__ __device__ float3 operator-(const float3 &a)
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

// Random
inline __host__ __device__ float pcg_rxs_m_xs(uint32_t* random_state)
{
    uint32_t state = *random_state;
    *random_state = *random_state * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (float)(((word >> 22u) ^ word) >> 8) * (1.0f / (UINT32_C(1) << 24));
}

//inline __host__ __device__ float pcg_xsh_rs(uint64_t* random_state)
//{
//	uint64_t old_state = *random_state;
//	*random_state = old_state * 6364136223846793005u;
//	old_state ^= old_state >> 22;
//	uint32_t word = (uint32_t)(old_state >> (22 + (uint32_t)(old_state >> 61)));
//	return (float)(word >> 8) * (1.0f / (UINT32_C(1) << 24));
//}
//
//inline __host__ __device__ void pcg_xsh_rs_init(const uint64_t seed, uint64_t* random_state)
//{
//	*random_state = 2 * seed + 1;
//	(void)pcg_xsh_rs(random_state);
//}
//
//static __host__ __device__ uint32_t rotr32(const uint32_t x, const uint32_t r)
//{
//	return x >> r | x << (-r & 31);
//}
//
//inline __host__ __device__ float pcg_xsh_rr(uint64_t* random_state)
//{
//	uint64_t old_state = *random_state;
//	*random_state = old_state * 6364136223846793005u + 1442695040888963407u;
//	old_state ^= old_state >> 18;
//	const uint32_t word = rotr32((uint32_t)(old_state >> 27), (uint32_t)(old_state >> 59));
//    return (float)(word >> 8) * (1.0f / (UINT32_C(1) << 24));
//}
//
//inline __host__ __device__ void pcg_xsh_rr_init(const uint64_t seed, uint64_t* random_state)
//{
//	*random_state = seed + 1442695040888963407u;
//	(void)pcg_xsh_rr(random_state);
//}

inline __host__ __device__ float3 disk_random(uint32_t* random_state)
{
	float3 v;
	do
	{
		v = 2.0f * make_float3(pcg_rxs_m_xs(random_state), pcg_rxs_m_xs(random_state), 0.0f) - make_float3(1.0f, 1.0f, 0.0f);
	} while (dot(v, v) >= 1.0f);
	return v;
}

inline __host__ __device__ float3 sphere_random(uint32_t* random_state)
{
	float3 v;
	do
	{
		v = make_float3(pcg_rxs_m_xs(random_state), pcg_rxs_m_xs(random_state), pcg_rxs_m_xs(random_state)) - make_float3(1.0f);
	} while (dot(v, v) >= 1.0f);
	return v;
}