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

	__host__ __device__ [[nodiscard]] bool intersect(const Ray& ray) const
	{
		float t0{kTMin}, t1{ray.t_max_};

		float inv_direction = 1.0f / ray.direction_.x;
		float t_near = (min_.x - ray.origin_.x) * inv_direction;
		float t_far = (max_.x - ray.origin_.x) * inv_direction;

        if (t_near > t_far)
        {
	        const float temp = t0;
			t0 = t1;
			t1 = temp;
        }

		t0 = t_near > t0 ? t_near : t0;
		t1 = t_far < t1 ? t_far : t1;

		if (t0 > t1)
			return false;

		inv_direction = 1.0f / ray.direction_.y;
		t_near = (min_.y - ray.origin_.y) * inv_direction;
		t_far = (max_.y - ray.origin_.y) * inv_direction;

        if (t_near > t_far)
        {
	        const float temp = t0;
			t0 = t1;
			t1 = temp;
        }

		t0 = t_near > t0 ? t_near : t0;
		t1 = t_far < t1 ? t_far : t1;

		if (t0 > t1)
			return false;

		inv_direction = 1.0f / ray.direction_.z;
		t_near = (min_.z - ray.origin_.z) * inv_direction;
		t_far = (max_.z - ray.origin_.z) * inv_direction;

        if (t_near > t_far)
        {
	        const float temp = t0;
			t0 = t1;
			t1 = temp;
        }

		t0 = t_near > t0 ? t_near : t0;
		t1 = t_far < t1 ? t_far : t1;

		if (t0 > t1)
			return false;
	    
		return true;
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
