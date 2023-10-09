#pragma once
#include "Math.hpp"

#include <cfloat>

class Ray
{
public:
    __host__ __device__ Ray(const float3 origin, const float3 direction)
		: origin_(origin), direction_(normalize(direction)), t_max_(FLT_MAX) {}

    float3 origin_;
    float3 direction_;
    mutable float t_max_;
};