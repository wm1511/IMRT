#pragma once
#include "Ray.cuh"

class Boundary
{
public:
	__host__ __device__ Boundary() : min_(make_float3(FLT_MAX)), max_(make_float3(-FLT_MAX)) {}
	__host__ __device__ explicit Boundary(const float3& p) : min_(p), max_(p) {}
	__host__ __device__ Boundary(const float3& p1, const float3& p2)
		: min_(make_float3(fminf(p1.x, p2.x), fminf(p1.y, p2.y), fminf(p1.z, p2.z))),
          max_(make_float3(fmaxf(p1.x, p2.x), fmaxf(p1.y, p2.y), fmaxf(p1.z, p2.z))) {}

	//A Ray-Box Intersection Algorithm and Efficient Dynamic Voxel Rendering
	__host__ __device__ [[nodiscard]] bool intersect(const Ray& ray) const
	{
		const float3 inverse_direction = 1.0f / ray.direction_;

		const float3 t_lower = (min_ - ray.origin_) * inverse_direction;
		const float3 t_upper = (max_ - ray.origin_) * inverse_direction;

		const float4 t_mins = make_float4(fminf(t_lower, t_upper), kTMin);
		const float4 t_maxes = make_float4(fmaxf(t_lower, t_upper), FLT_MAX);

		const float t_boundary_min = maxcomp(t_mins);
		const float t_boundary_max = mincomp(t_maxes);
		return t_boundary_min <= t_boundary_max;
	}

	float3 min_, max_;
};

inline __host__ __device__ Boundary unite(const Boundary& b1, const Boundary& b2)
{
	return {fminf(b1.min_, b2.min_), fmaxf(b1.max_, b2.max_)};
}

inline __host__ __device__ Boundary unite(const Boundary& b1, const float3& v)
{
	return {fminf(b1.min_, v), fmaxf(b1.max_, v)};
}
