#pragma once
#include <cuda_runtime.h>

#ifndef __CUDACC__
#include <cmath>

inline float fminf(const float a, const float b)
{
	return a < b ? a : b;
}

inline float fmaxf(const float a, const float b)
{
	return a > b ? a : b;
}

inline float rsqrtf(const float x)
{
	return 1.0f / sqrtf(x);
}

#endif

// Constants
__device__ __constant__ constexpr float kPi = 3.141593f;
__device__ __constant__ constexpr float k2Pi = 6.283185f;
__device__ __constant__ constexpr float kHalfPi =  1.570796f;
__device__ __constant__ constexpr float kInvPi = 0.318309f;
__device__ __constant__ constexpr float kInv2Pi = 0.159154f;
__device__ __constant__ constexpr float kTMin = 0.001f;

// Constructors

inline __host__ __device__ float3 make_float3(const float s)
{
    return make_float3(s, s, s);
}

inline __host__ __device__ float3 make_float3(const float t[3])
{
	return make_float3(t[0], t[1], t[2]);
}

inline __host__ __device__ float4 make_float4(const float3 v, const float s)
{
    return make_float4(v.x, v.y, v.z, s);
}

inline __host__ __device__ float4 make_float4(const float t[4])
{
	return make_float4(t[0], t[1], t[2], t[3]);
}

// Negation
inline __host__ __device__ float2 operator-(float2 &a)
{
    return make_float2(-a.x, -a.y);
}

inline __host__ __device__ float3 operator-(float3 &a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

inline __host__ __device__ float3 operator-(const float3 &a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

// Addition
inline __host__ __device__ float2 operator+(const float2 a, const float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(float2 &a, const float2 b)
{
    a.x += b.x;
    a.y += b.y;
}
inline __host__ __device__ float2 operator+(const float2 a, const float b)
{
    return make_float2(a.x + b, a.y + b);
}
inline __host__ __device__ float2 operator+(const float b, const float2 a)
{
    return make_float2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(float2 &a, const float b)
{
    a.x += b;
    a.y += b;
}
inline __host__ __device__ float3 operator+(const float3 a, const float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ float4 operator+(const float4 a, const float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ void operator+=(float3 &a, const float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ void operator+=(float4 &a, const float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline __host__ __device__ float3 operator+(const float3 a, const float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ float4 operator+(const float4 a, const float b)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ void operator+=(float3 &a, const float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}
inline __host__ __device__ void operator+=(float4 &a, const float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}
inline __host__ __device__ float3 operator+(const float b, const float3 a)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ float4 operator+(const float b, const float4 a)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}

// Subtraction
inline __host__ __device__ float2 operator-(const float2 a, const float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(float2 &a, const float2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline __host__ __device__ float2 operator-(const float2 a, const float b)
{
    return make_float2(a.x - b, a.y - b);
}
inline __host__ __device__ float2 operator-(const float b, const float2 a)
{
    return make_float2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(float2 &a, const float b)
{
    a.x -= b;
    a.y -= b;
}
inline __host__ __device__ float3 operator-(const float3 a, const float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ float4 operator-(const float4 a, const float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline __host__ __device__ void operator-=(float3 &a, const float3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __host__ __device__ void operator-=(float4 &a, const float4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline __host__ __device__ float3 operator-(const float3 a, const float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ float4 operator-(const float4 a, const float b)
{
    return make_float4(a.x - b, a.y - b, a.z - b, a.w - b);
}
inline __host__ __device__ void operator-=(float3 &a, const float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}
inline __host__ __device__ void operator-=(float4 &a, const float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}
inline __host__ __device__ float3 operator-(const float b, const float3 a)
{
    return make_float3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ float4 operator-(const float b, const float4 a)
{
	return make_float4(a.x - b, a.y - b, a.z - b, a.w - b);
}

// Multiplication
inline __host__ __device__ float2 operator*(const float2 a, const float2 b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(float2 &a, const float2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
inline __host__ __device__ float2 operator*(const float2 a, const float b)
{
    return make_float2(a.x * b, a.y * b);
}
inline __host__ __device__ float2 operator*(const float b, const float2 a)
{
    return make_float2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(float2 &a, const float b)
{
    a.x *= b;
    a.y *= b;
}
inline __host__ __device__ float3 operator*(const float3 a, const float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ float4 operator*(const float4 a, const float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __host__ __device__ void operator*=(float3 &a, const float3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline __host__ __device__ void operator*=(float4 &a, const float4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline __host__ __device__ float3 operator*(const float3 a, const float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ float4 operator*(const float4 a, const float b)
{
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}
inline __host__ __device__ void operator*=(float3 &a, const float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}
inline __host__ __device__ void operator*=(float4 &a, const float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}
inline __host__ __device__ float3 operator*(const float b, const float3 a)
{
    return make_float3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ float4 operator*(const float b, const float4 a)
{
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

// Division
inline __host__ __device__ float2 operator/(const float2 a, const float2 b)
{
    return make_float2(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ void operator/=(float2 &a, const float2 b)
{
    a.x /= b.x;
    a.y /= b.y;
}
inline __host__ __device__ float2 operator/(const float2 a, const float b)
{
    return make_float2(a.x / b, a.y / b);
}
inline __host__ __device__ void operator/=(float2 &a, const float b)
{
    a.x /= b;
    a.y /= b;
}
inline __host__ __device__ float2 operator/(const float b, const float2 a)
{
    return make_float2(b / a.x, b / a.y);
}
inline __host__ __device__ float3 operator/(const float3 a, const float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ float4 operator/(const float4 a, const float4 b)
{
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
inline __host__ __device__ void operator/=(float3 &a, const float3 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}
inline __host__ __device__ void operator/=(float4 &a, const float4 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
}
inline __host__ __device__ float3 operator/(const float3 a, const float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ float4 operator/(const float4 a, const float b)
{
    return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}
inline __host__ __device__ void operator/=(float3 &a, const float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
}
inline __host__ __device__ void operator/=(float4 &a, const float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}
inline __host__ __device__ float3 operator/(const float b, const float3 a)
{
    return make_float3(b / a.x, b / a.y, b / a.z);
}
inline __host__ __device__ float4 operator/(const float b, const float4 a)
{
    return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}

// Minimum
inline  __host__ __device__ float2 fminf(const float2 a, const float2 b)
{
    return make_float2(fminf(a.x,b.x), fminf(a.y,b.y));
}
inline  __host__ __device__ float2 fminf(const float2 a, const float2 b, const float2 c)
{
    return make_float2(fminf(c.x, fminf(a.x,b.x)), fminf(c.y, fminf(a.y,b.y)));
}
inline __host__ __device__ float3 fminf(const float3 a, const float3 b)
{
    return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}
inline  __host__ __device__ float3 fminf(const float3 a, const float3 b, const float3 c)
{
    return make_float3(fminf(c.x, fminf(a.x,b.x)), fminf(c.y, fminf(a.y,b.y)), fminf(c.z, fminf(a.z,b.z)));
}
inline  __host__ __device__ float4 fminf(const float4 a, const float4 b)
{
    return make_float4(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z), fminf(a.w,b.w));
}
inline  __host__ __device__ float4 fminf(const float4 a, const float4 b, const float4 c)
{
    return make_float4(fminf(c.x, fminf(a.x,b.x)), fminf(c.y, fminf(a.y,b.y)), fminf(c.z, fminf(a.z,b.z)), fminf(c.w, fminf(a.w,b.w)));
}

// Maximum
inline __host__ __device__ float2 fmaxf(const float2 a, const float2 b)
{
    return make_float2(fmaxf(a.x,b.x), fmaxf(a.y,b.y));
}
inline  __host__ __device__ float2 fmaxf(const float2 a, const float2 b, const float2 c)
{
    return make_float2(fmaxf(c.x, fmaxf(a.x,b.x)), fmaxf(c.y, fmaxf(a.y,b.y)));
}
inline __host__ __device__ float3 fmaxf(const float3 a, const float3 b)
{
    return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}
inline  __host__ __device__ float3 fmaxf(const float3 a, const float3 b, const float3 c)
{
    return make_float3(fmaxf(c.x, fmaxf(a.x,b.x)), fmaxf(c.y, fmaxf(a.y,b.y)), fmaxf(c.z, fmaxf(a.z,b.z)));
}
inline __host__ __device__ float4 fmaxf(const float4 a, const float4 b)
{
    return make_float4(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z), fmaxf(a.w,b.w));
}
inline  __host__ __device__ float4 fmaxf(const float4 a, const float4 b, const float4 c)
{
    return make_float4(fmaxf(c.x, fmaxf(a.x,b.x)), fmaxf(c.y, fmaxf(a.y,b.y)), fmaxf(c.z, fmaxf(a.z,b.z)), fmaxf(c.w, fmaxf(a.w,b.w)));
}

// Linear interpolation
inline __device__ __host__ float lerp(const float a, const float b, const float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float2 lerp(const float2 a, const float2 b, const float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float3 lerp(const float3 a, const float3 b, const float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float4 lerp(const float4 a, const float4 b, const float t)
{
    return a + t*(b-a);
}

// Fraction
inline __host__ __device__ float fracf(const float v)
{
    return v - floorf(v);
}
inline __host__ __device__ float2 fracf(const float2 v)
{
    return make_float2(fracf(v.x), fracf(v.y));
}
inline __host__ __device__ float3 fracf(const float3 v)
{
    return make_float3(fracf(v.x), fracf(v.y), fracf(v.z));
}
inline __host__ __device__ float4 fracf(const float4 v)
{
    return make_float4(fracf(v.x), fracf(v.y), fracf(v.z), fracf(v.w));
}

// Clamp
inline __host__ __device__ float clamp(const float f, const float a, const float b)
{
    return fmaxf(a, fminf(f, b));
}
inline __host__ __device__ float3 clamp(const float3 v, const float a, const float b)
{
    return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __host__ __device__ float3 clamp(const float3 v, const float3 a, const float3 b)
{
    return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}

// Dot product
inline __host__ __device__ float dot(const float2 a, const float2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ float dot(const float3 a, const float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ float dot(const float4 a, const float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// Length
inline __host__ __device__ float length(const float2 v)
{
    return sqrtf(dot(v, v));
}
inline __host__ __device__ float length(const float3 v)
{
    return sqrtf(dot(v, v));
}
inline __host__ __device__ float length(const float4 v)
{
    return sqrtf(dot(v, v));
}


// Normalization
inline __host__ __device__ float3 normalize(const float3 v)
{
    return v * rsqrtf(dot(v, v));
}

// Floor
inline __host__ __device__ float3 floor(const float3 v)
{
    return make_float3(floorf(v.x), floorf(v.y), floorf(v.z));
}

// Ceil
inline __host__ __device__ float3 ceil(const float3 v)
{
    return make_float3(ceilf(v.x), ceilf(v.y), ceilf(v.z));
}

// Floor
inline __host__ __device__ float3 round(const float3 v)
{
    return make_float3(roundf(v.x), roundf(v.y), roundf(v.z));
}

// Absolute value
inline __host__ __device__ float3 abs(const float3 v)
{
    return make_float3(fabs(v.x), fabs(v.y), fabs(v.z));
}

// Square root
inline __host__ __device__ float3 sqrt(const float3 v)
{
	return make_float3(sqrt(v.x), sqrt(v.y), sqrt(v.z));
}

inline __host__ __device__ float4 sqrt(const float4 v)
{
	return make_float4(sqrt(v.x), sqrt(v.y), sqrt(v.z), sqrt(v.w));
}

// Reflection
inline __host__ __device__ float3 reflect(const float3 i, const float3 n)
{
    return i - 2.0f * n * dot(n,i);
}

// Cross product
inline __host__ __device__ float3 cross(const float3 a, const float3 b)
{
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

// Versor
inline __host__ __device__ float3 versor(const float3 v)
{
	return float3(v / length(v));
}

// Matrix 4x4
union Matrix4X4
{
    float m[4][4]{};

private:
    struct _
    {
	    float4 x{1.0f, 0.0f, 0.0f, 0.0f};
    	float4 y{0.0f, 1.0f, 0.0f, 0.0f};
    	float4 z{0.0f, 0.0f, 1.0f, 0.0f};
    	float4 w{0.0f, 0.0f, 0.0f, 1.0f};
    };
};

inline __host__ __device__ Matrix4X4 make_matrix4X4(const float s00, const float s01, const float s02, const float s03, 
    const float s10, const float s11, const float s12, const float s13, const float s20, const float s21, 
    const float s22, const float s23, const float s30, const float s31, const float s32, const float s33)
{
    return
    {
	    {
		    {s00, s01, s02, s03},
		    {s10, s11, s12, s13},
		    {s20, s21, s22, s23},
		    {s30, s31, s32, s33}
	    }
    };
}

inline __host__ __device__ Matrix4X4 make_matrix4X4()
{
    return {};
}

inline __host__ __device__ Matrix4X4 transpose(const Matrix4X4& m)
{
    return
    {
	    {
		    {m.m[0][0], m.m[1][0], m.m[2][0], m.m[3][0]},
		    {m.m[0][1], m.m[1][1], m.m[2][1], m.m[3][1]},
		    {m.m[0][2], m.m[1][2], m.m[2][2], m.m[3][2]},
		    {m.m[0][3], m.m[1][3], m.m[2][3], m.m[3][3]}
	    }
    };
}
    
inline __host__ __device__ Matrix4X4 multiply(const Matrix4X4 &m1, const Matrix4X4 &m2)
{
    Matrix4X4 r;
    for (int32_t i = 0; i < 4; i++)
	    for (int32_t j = 0; j < 4; j++)
		    r.m[i][j] = m1.m[i][0] * m2.m[0][j] + m1.m[i][1] * m2.m[1][j] +
			    m1.m[i][2] * m2.m[2][j] + m1.m[i][3] * m2.m[3][j];
    return r;
}

inline __host__ __device__ Matrix4X4 invert(const Matrix4X4& m)
{
	const float a2323 = m.m[2][2] * m.m[3][3] - m.m[2][3] * m.m[3][2];
	const float a1323 = m.m[2][1] * m.m[3][3] - m.m[2][3] * m.m[3][1];
	const float a1223 = m.m[2][1] * m.m[3][2] - m.m[2][2] * m.m[3][1];
	const float a0323 = m.m[2][0] * m.m[3][3] - m.m[2][3] * m.m[3][0];
	const float a0223 = m.m[2][0] * m.m[3][2] - m.m[2][2] * m.m[3][0];
	const float a0123 = m.m[2][0] * m.m[3][1] - m.m[2][1] * m.m[3][0];
	const float a2313 = m.m[1][2] * m.m[3][3] - m.m[1][3] * m.m[3][2];
	const float a1313 = m.m[1][1] * m.m[3][3] - m.m[1][3] * m.m[3][1];
	const float a1213 = m.m[1][1] * m.m[3][2] - m.m[1][2] * m.m[3][1];
	const float a2312 = m.m[1][2] * m.m[2][3] - m.m[1][3] * m.m[2][2];
	const float a1312 = m.m[1][1] * m.m[2][3] - m.m[1][3] * m.m[2][1];
	const float a1212 = m.m[1][1] * m.m[2][2] - m.m[1][2] * m.m[2][1];
	const float a0313 = m.m[1][0] * m.m[3][3] - m.m[1][3] * m.m[3][0];
	const float a0213 = m.m[1][0] * m.m[3][2] - m.m[1][2] * m.m[3][0];
	const float a0312 = m.m[1][0] * m.m[2][3] - m.m[1][3] * m.m[2][0];
	const float a0212 = m.m[1][0] * m.m[2][2] - m.m[1][2] * m.m[2][0];
	const float a0113 = m.m[1][0] * m.m[3][1] - m.m[1][1] * m.m[3][0];
	const float a0112 = m.m[1][0] * m.m[2][1] - m.m[1][1] * m.m[2][0];

    float det = m.m[0][0] * ( m.m[1][1] * a2323 - m.m[1][2] * a1323 + m.m[1][3] * a1223 )
        - m.m[0][1] * ( m.m[1][0] * a2323 - m.m[1][2] * a0323 + m.m[1][3] * a0223 )
        + m.m[0][2] * ( m.m[1][0] * a1323 - m.m[1][1] * a0323 + m.m[1][3] * a0123 )
        - m.m[0][3] * ( m.m[1][0] * a1223 - m.m[1][1] * a0223 + m.m[1][2] * a0123 );
    det = 1.0f / det;

	return
	{
		{
			{
				det * (m.m[1][1] * a2323 - m.m[1][2] * a1323 + m.m[1][3] * a1223),
				det * -(m.m[0][1] * a2323 - m.m[0][2] * a1323 + m.m[0][3] * a1223),
				det * (m.m[0][1] * a2313 - m.m[0][2] * a1313 + m.m[0][3] * a1213),
				det * -(m.m[0][1] * a2312 - m.m[0][2] * a1312 + m.m[0][3] * a1212)
			},
			{
				det * -(m.m[1][0] * a2323 - m.m[1][2] * a0323 + m.m[1][3] * a0223),
				det * (m.m[0][0] * a2323 - m.m[0][2] * a0323 + m.m[0][3] * a0223),
				det * -(m.m[0][0] * a2313 - m.m[0][2] * a0313 + m.m[0][3] * a0213),
				det * (m.m[0][0] * a2312 - m.m[0][2] * a0312 + m.m[0][3] * a0212)
			},
			{
				det * (m.m[1][0] * a1323 - m.m[1][1] * a0323 + m.m[1][3] * a0123),
				det * -(m.m[0][0] * a1323 - m.m[0][1] * a0323 + m.m[0][3] * a0123),
				det * (m.m[0][0] * a1313 - m.m[0][1] * a0313 + m.m[0][3] * a0113),
				det * -(m.m[0][0] * a1312 - m.m[0][1] * a0312 + m.m[0][3] * a0112)
			},
			{
				det * -(m.m[1][0] * a1223 - m.m[1][1] * a0223 + m.m[1][2] * a0123),
				det * (m.m[0][0] * a1223 - m.m[0][1] * a0223 + m.m[0][2] * a0123),
				det * -(m.m[0][0] * a1213 - m.m[0][1] * a0213 + m.m[0][2] * a0113),
				det * (m.m[0][0] * a1212 - m.m[0][1] * a0212 + m.m[0][2] * a0112)
			}
		}
	};
}

// Transformations
inline __host__ __device__ void translate_point(float3& v, const float3 t)
{
    v += t;
}

inline __host__ __device__ void scale_point(float3& v, const float3 s)
{
	v *= s;
}

inline __host__ __device__ void rotate_point_x(float3& v, const float rx)
{
	const float sc = sin(rx); 
	const float cc = cos(rx);

    v = {v.x, cc * v.y - sc * v.z, sc * v.y + cc * v.z};
}

inline __host__ __device__ void rotate_point_y(float3& v, const float ry)
{
	const float sb = sin(ry); 
	const float cb = cos(ry);

    v = {cb * v.x - sb * v.z, v.y, sb * v.x + cb * v.z};
}

inline __host__ __device__ void rotate_point_z(float3& v, const float rz)
{
	const float sa = sin(rz); 
	const float ca = cos(rz);

    v = {ca * v.x - sa * v.y, sa * v.x + ca * v.y, v.z};
}

inline __host__ __device__ void transform_point(float3& v, const float3 t, const float3 s, const float3 r)
{
    translate_point(v, t);

    scale_point(v, s);

	if (r.x > 0.0f)
		rotate_point_x(v, r.x);

	if (r.y > 0.0f)
		rotate_point_y(v, r.y);

	if (r.z > 0.0f)
		rotate_point_z(v, r.z);
}

// Random
inline __host__ __device__ uint32_t rotl(const uint32_t x, const int k)
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