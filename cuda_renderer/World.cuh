#pragma once
#include "Primitive.cuh"

class World final : public Primitive
{
public:
	__device__ explicit World(Primitive** primitives, const uint32_t primitive_count) : primitives_(primitives), primitive_count_(primitive_count) {}
	__device__ bool intersect(const Ray& ray, float t_min, float t_max, Intersection& intersection) const override
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

private:
	Primitive** primitives_;
	uint32_t primitive_count_;
};