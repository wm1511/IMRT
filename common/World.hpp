#pragma once
#include "../info/Texture.hpp"
#include "../info/Material.hpp"
#include "../info/Object.hpp"

class World
{
public:
	__host__ __device__ World(Object* object, Material* material, Texture* texture, const int32_t object_count, const int32_t material_count, const int32_t texture_count)
		: objects_(object), materials_(material), textures_(texture), object_count_(object_count), material_count_(material_count), texture_count_(texture_count) {}

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
	int32_t object_count_, material_count_, texture_count_;
};
