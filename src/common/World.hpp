#pragma once
#include "../common/Texture.hpp"
#include "../common/Material.hpp"
#include "../common/Object.hpp"

class World
{
public:
	__host__ __device__ World(Object* object, Material* material, Texture* texture, const int32_t object_count)
		: objects_(object), materials_(material), textures_(texture), object_count_(object_count) {}

	__host__ __device__ bool intersect(const Ray& ray, Intersection& intersection) const
	{
		Intersection temp_intersection{};
		bool intersected = false;

		for (int32_t i = 0; i < object_count_; i++)
		{
			if (objects_[i].bound().intersect(ray))
			{
				if (objects_[i].intersect(ray, temp_intersection))
				{
					intersected = true;
					intersection = temp_intersection;
					intersection.texture = &textures_[objects_[i].texture_id];
					intersection.material = &materials_[objects_[i].material_id];
				}
			}
		}

		return intersected;
	}

private:
	Object* objects_;
	Material* materials_;
	Texture* textures_;
	int32_t object_count_
;
};
