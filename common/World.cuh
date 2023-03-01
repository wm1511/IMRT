#pragma once
#include "Material.cuh"
#include "Object.cuh"

class World
{
public:
	__host__ __device__ World(ObjectInfo** object_data, MaterialInfo** material_data, const int32_t object_count, const int32_t material_count) : object_count_(object_count), material_count_(material_count)
	{
		materials_ = new Material*[material_count];
		objects_ = new Object*[object_count];

		for (int32_t i = 0; i < material_count; i++)
		{
			const auto material_info = material_data[i];

			if (material_info->type == DIFFUSE)
				materials_[i] = new Diffuse((DiffuseInfo*)material_info);
			else if (material_info->type == SPECULAR)
				materials_[i] = new Specular((SpecularInfo*)material_info);
			else if (material_info->type == REFRACTIVE)
				materials_[i] = new Refractive((RefractiveInfo*)material_info);
			else if (material_info->type == ISOTROPIC)
				materials_[i] = new Isotropic((IsotropicInfo*)material_info);
			else if (material_info->type == TEXTURE)
				materials_[i] = new Texture((TextureInfo*)material_info);
		}

		for (int32_t i = 0; i < object_count; i++)
		{
			const auto object_info = object_data[i];

			if (object_info->type == SPHERE)
				objects_[i] = new Sphere((SphereInfo*)object_info, materials_[object_info->material_id]);
			else if (object_info->type == TRIANGLE)
				objects_[i] = new Triangle((TriangleInfo*)object_info, materials_[object_info->material_id]);
			else if (object_info->type == PLANE)
				objects_[i] = new Plane((PlaneInfo*)object_info, materials_[object_info->material_id]);
			else if (object_info->type == VOLUMETRIC_SPHERE)
				objects_[i] = new VolumetricSphere((VolumetricSphereInfo*)object_info, materials_[object_info->material_id]);
			else if (object_info->type == CYLINDER)
				objects_[i] = new Cylinder((CylinderInfo*)object_info, materials_[object_info->material_id]);
			else if (object_info->type == CONE)
				objects_[i] = new Cone((ConeInfo*)object_info, materials_[object_info->material_id]);
			else if (object_info->type == TORUS)
				objects_[i] = new Torus((TorusInfo*)object_info, materials_[object_info->material_id]);
			else if (object_info->type == MODEL)
				objects_[i] = new Model((ModelInfo*)object_info, materials_[object_info->material_id]);
		}
	}

	__host__ __device__ ~World()
	{
		for (int32_t i = 0; i < material_count_; i++)
			delete materials_[i];

		for (int32_t i = 0; i < object_count_; i++)
			delete objects_[i];

		delete[] objects_;
		delete[] materials_;
	}

	__host__ __device__ bool intersect(const Ray& ray, const float t_min, const float t_max, Intersection& intersection, uint32_t* random_state) const
	{
		Intersection temp_intersection{};
		bool intersected = false;
		float potentially_closest = t_max;

		for (int32_t i = 0; i < object_count_; i++)
		{
			if (objects_[i]->intersect(ray, t_min, potentially_closest, temp_intersection, random_state))
			{
				intersected = true;
				potentially_closest = temp_intersection.t;
				intersection = temp_intersection;
			}
		}

		return intersected;
	}

	__host__ __device__ void update_material(const int32_t index, MaterialInfo* material_info) const
	{
		materials_[index]->update(material_info);
	}

	__host__ __device__ void update_object(const int32_t index, ObjectInfo* object_info) const
	{
		objects_[index]->update(object_info, materials_[object_info->material_id]);
	}

private:
	Object** objects_;
	Material** materials_;
	int32_t object_count_, material_count_;
};
