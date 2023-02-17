#pragma once
#include "Primitive.cuh"

class World
{
public:
	__host__ __device__ explicit World(Primitive** primitives, const uint32_t primitive_count) : primitives_(primitives), primitive_count_(primitive_count) {}

	__host__ __device__ bool intersect(const Ray& ray, const float t_min, const float t_max, Intersection& intersection) const
	{
		Intersection temp_intersection{};
		bool intersected = false;
		float potentially_closest = t_max;
		for (uint32_t i = 0; i < primitive_count_; i++)
		{
			if (primitives_[i]->intersect(ray, t_min, potentially_closest, temp_intersection))
			{
				intersected = true;
				potentially_closest = temp_intersection.t;
				intersection = temp_intersection;
			}
		}

		return intersected;
	}

	__host__ __device__ void update(const uint32_t primitive_count)
	{
		primitive_count_ = primitive_count;
	}

private:
	Primitive** primitives_;
	uint32_t primitive_count_;
};
