#pragma once
#include "Texture.cuh"
#include "Material.cuh"
#include "Object.cuh"

class World
{
public:
	__host__ __device__ World(ObjectInfo** object_data, MaterialInfo** material_data, TextureInfo** texture_data, const int32_t object_count, const int32_t material_count, const int32_t texture_count)
		: object_count_(object_count), material_count_(material_count), texture_count_(texture_count)
	{
		textures_ = new Texture*[texture_count];
		materials_ = new Material*[material_count];
		objects_ = new Object*[object_count];

		for (int32_t i = 0; i < texture_count; i++)
		{
			const auto texture_info = texture_data[i];

			if (texture_info->type == SOLID)
				textures_[i] = new Solid((SolidInfo*)texture_info);
			else if (texture_info->type == IMAGE)
				textures_[i] = new Image((ImageInfo*)texture_info);
			else if (texture_info->type == CHECKER)
				textures_[i] = new Checker((CheckerInfo*)texture_info);
		}

		for (int32_t i = 0; i < material_count; i++)
		{
			const auto material_info = material_data[i];

			if (material_info->type == DIFFUSE)
				materials_[i] = new Diffuse((DiffuseInfo*)material_info, textures_[material_info->texture_id]);
			else if (material_info->type == SPECULAR)
				materials_[i] = new Specular((SpecularInfo*)material_info, textures_[material_info->texture_id]);
			else if (material_info->type == REFRACTIVE)
				materials_[i] = new Refractive((RefractiveInfo*)material_info, textures_[material_info->texture_id]);
			else if (material_info->type == ISOTROPIC)
				materials_[i] = new Isotropic((IsotropicInfo*)material_info, textures_[material_info->texture_id]);
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
		for (int32_t i = 0; i < texture_count_; i++)
			delete textures_[i];

		for (int32_t i = 0; i < material_count_; i++)
			delete materials_[i];

		for (int32_t i = 0; i < object_count_; i++)
			delete objects_[i];

		delete[] objects_;
		delete[] materials_;
		delete[] textures_;
	}

	__host__ __device__ bool intersect(const Ray& ray, Intersection& intersection, uint32_t* random_state) const
	{
		Intersection temp_intersection{};
		bool intersected = false;

		for (int32_t i = 0; i < object_count_; i++)
		{
			//if (objects_[i]->bound().intersect(ray))
			
			if (objects_[i]->intersect(ray, temp_intersection, random_state))
			{
				intersected = true;
				intersection = temp_intersection;
			}
			
		}

		return intersected;
	}

	__host__ __device__ void update_texture(const int32_t index, TextureInfo* texture_info) const
	{
		textures_[index]->update(texture_info);
	}

	__host__ __device__ void update_material(const int32_t index, MaterialInfo* material_info) const
	{
		materials_[index]->update(material_info, textures_[material_info->texture_id]);
	}

	__host__ __device__ void update_object(const int32_t index, ObjectInfo* object_info) const
	{
		objects_[index]->update(object_info, materials_[object_info->material_id]);
	}

private:
	Object** objects_;
	Material** materials_;
	Texture** textures_;
	int32_t object_count_, material_count_, texture_count_;
};
