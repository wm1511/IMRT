#pragma once
#include "Math.cuh"

class Ray
{
public:
    __host__ __device__ Ray(const float3 origin, const float3 direction) : origin_(origin), direction_(direction) {}
    __host__ __device__ [[nodiscard]] float3 origin() const
    {
	    return origin_;
    }
    __host__ __device__ [[nodiscard]] float3 direction() const
    {
	    return direction_;
    }
    __host__ __device__ [[nodiscard]] float3 position(const float t) const
    {
	    return origin_ + t * direction_;
    }

private:
    float3 origin_;
    float3 direction_;
};