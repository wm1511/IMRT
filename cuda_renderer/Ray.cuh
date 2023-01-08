#pragma once
#include "Math.cuh"

class Ray
{
public:
    __device__ Ray(const float3& origin, const float3& direction) : origin_(origin), direction_(direction) {}
    __device__ float3 origin() const { return origin_; }
    __device__ float3 direction() const { return direction_; }
    __device__ float3 position(const float t) const { return origin_ + t * direction_; }

private:
    float3 origin_;
    float3 direction_;
};