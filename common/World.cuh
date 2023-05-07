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
			textures_[i] = create_texture(texture_data[i]);

		for (int32_t i = 0; i < material_count; i++)
			materials_[i] = create_material(material_data[i]);

		for (int32_t i = 0; i < object_count; i++)
			objects_[i] = create_object(object_data[i]);
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

	__host__ __device__ World(const World&) = delete;
	__host__ __device__ World(World&&) = delete;
	__host__ __device__ World& operator=(const World&) = delete;
	__host__ __device__ World& operator=(World&&) = delete;

	__host__ __device__ bool intersect(const Ray& ray, Intersection& intersection) const
	{
		Intersection temp_intersection{};
		bool intersected = false;

		for (int32_t i = 0; i < object_count_; i++)
		{
			if (objects_[i]->bound().intersect(ray))
			{
				if (objects_[i]->intersect(ray, temp_intersection))
				{
					intersected = true;
					intersection = temp_intersection;
				}
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

	__host__ __device__ Texture* create_texture(TextureInfo* texture_info) const
	{
		if (texture_info->type == TextureType::SOLID)
			return new Solid((SolidInfo*)texture_info);
		if (texture_info->type == TextureType::IMAGE)
			return new Image((ImageInfo*)texture_info);
		if (texture_info->type == TextureType::CHECKER)
			return new Checker((CheckerInfo*)texture_info);
		return nullptr;
	}

	__host__ __device__ Material* create_material(MaterialInfo* material_info) const
	{
		if (material_info->type == MaterialType::DIFFUSE)
			return new Diffuse((DiffuseInfo*)material_info, textures_[material_info->texture_id]);
		if (material_info->type == MaterialType::SPECULAR)
			return new Specular((SpecularInfo*)material_info, textures_[material_info->texture_id]);
		if (material_info->type == MaterialType::REFRACTIVE)
			return new Refractive((RefractiveInfo*)material_info, textures_[material_info->texture_id]);
		if (material_info->type == MaterialType::ISOTROPIC)
			return new Isotropic((IsotropicInfo*)material_info, textures_[material_info->texture_id]);
		return nullptr;
	}

	__host__ __device__ Object* create_object(ObjectInfo* object_info) const
	{
		if (object_info->type == ObjectType::SPHERE)
			return new Sphere((SphereInfo*)object_info, materials_[object_info->material_id]);
		if (object_info->type == ObjectType::PLANE)
			return new Plane((PlaneInfo*)object_info, materials_[object_info->material_id]);
		if (object_info->type == ObjectType::CYLINDER)
			return new Cylinder((CylinderInfo*)object_info, materials_[object_info->material_id]);
		if (object_info->type == ObjectType::CONE)
			return new Cone((ConeInfo*)object_info, materials_[object_info->material_id]);
		if (object_info->type == ObjectType::MODEL)
			return new Model((ModelInfo*)object_info, materials_[object_info->material_id]);
		return nullptr;
	}

private:
	Object** objects_;
	Material** materials_;
	Texture** textures_;
	int32_t object_count_, material_count_, texture_count_;
};
