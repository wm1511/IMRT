#pragma once
#include "Math.cuh"

class Ray
{
public:
    __host__ __device__ Ray(const float3 origin, const float3 direction, const float t_max = FLT_MAX)
		: origin_(origin), direction_(direction), t_max_(t_max) {}

    __host__ __device__ [[nodiscard]] float3 position(const float t) const
    {
	    return origin_ + t * direction_;
    }

    float3 origin_;
    float3 direction_;
    mutable float t_max_;
};