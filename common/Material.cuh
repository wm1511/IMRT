#pragma once
#include "Intersection.cuh"
#include "Ray.cuh"
#include "Texture.cuh"
#include "../info/MaterialInfo.hpp"

class Material
{
public:
	__host__ __device__ virtual ~Material() {}

	__host__ __device__ virtual bool scatter(Ray& ray, const Intersection& intersection, float3& absorption, uint32_t* random_state) const = 0;
	__host__ __device__ virtual void update(MaterialInfo* material_info, Texture* texture) = 0;

	Texture* texture_ = nullptr;
};

class Diffuse final : public Material
{
public:
	__host__ __device__ explicit Diffuse(const DiffuseInfo*, Texture* texture)
	{
		texture_ = texture;
	}

	__host__ __device__ bool scatter(Ray& ray, const Intersection& intersection, float3& absorption, uint32_t* random_state) const override
	{
		const float3 reflected_direction = intersection.normal + sphere_random(random_state);
		ray = Ray(intersection.point, reflected_direction);
		absorption = texture_->color(intersection.uv);
		return true;
	}

	__host__ __device__ void update(MaterialInfo*, Texture* texture) override
	{
		texture_ = texture;
	}
};

class Specular final : public Material
{
public:
	__host__ __device__ explicit Specular(const SpecularInfo* specular_info, Texture* texture)
		: fuzziness_(specular_info->fuzziness)
	{
		texture_ = texture;
	}

	__host__ __device__ bool scatter(Ray& ray, const Intersection& intersection, float3& absorption, uint32_t* random_state) const override
	{
		const float3 reflected_direction = reflect(versor(ray.direction_), intersection.normal);
		ray = Ray(intersection.point, reflected_direction + fuzziness_ * sphere_random(random_state));
		absorption = texture_->color(intersection.uv);
		return dot(ray.direction_, intersection.normal) > 0.0f;
	}

	__host__ __device__ void update(MaterialInfo* material, Texture* texture) override
	{
		const SpecularInfo* specular_info = (SpecularInfo*)material;
		fuzziness_ = specular_info->fuzziness;
		texture_ = texture;
	}

private:
	float fuzziness_{};
};

class Refractive final : public Material
{
public:
	__host__ __device__ explicit Refractive(const RefractiveInfo* refractive_info, Texture* texture)
		: refractive_index_(refractive_info->refractive_index)
	{
		texture_ = texture;
	}

	__host__ __device__ bool scatter(Ray& ray, const Intersection& intersection, float3& absorption, uint32_t* random_state) const override
	{
		bool refracted;
		float ior;
		float reflection_probability;
		float cos_theta;
		float3 normal_out;
		float3 refracted_direction{};
		const float3 reflected_direction = reflect(ray.direction_, intersection.normal);
		absorption = make_float3(1.0f);

		if (dot(ray.direction_, intersection.normal) > 0.0f)
		{
			normal_out = -intersection.normal;
			ior = refractive_index_;
			cos_theta = dot(ray.direction_, intersection.normal) / length(ray.direction_);
			cos_theta = sqrt(1.0f - refractive_index_ * refractive_index_ * (1.0f - cos_theta * cos_theta));
		}
		else
		{
			normal_out = intersection.normal;
			ior = 1.0f / refractive_index_;
			cos_theta = -dot(ray.direction_, intersection.normal) / length(ray.direction_);
		}

		const float3 unit_ray_direction = versor(ray.direction_);
		const float ray_normal_dot = dot(unit_ray_direction, normal_out);
		const float discriminant = 1.0f - ior * ior * (1 - ray_normal_dot * ray_normal_dot);
		if (discriminant > 0)
		{
			refracted_direction = ior * (unit_ray_direction - normal_out * ray_normal_dot) - normal_out * sqrt(discriminant);
			refracted = true;
		}
		else
		{
			refracted = false;
		}

		if (refracted)
		{
			float r0 = (1.0f - refractive_index_) / (1.0f + refractive_index_);
			r0 = r0 * r0;
			reflection_probability = r0 + (1.0f - r0) * pow((1.0f - cos_theta), 5.0f);
		}
		else
		{
			reflection_probability = 1.0f;
		}

		if (pcg(random_state) < reflection_probability)
		{
			ray = Ray(intersection.point, reflected_direction);
		}
		else
		{
			ray =  Ray(intersection.point, refracted_direction);
		}
		return true;
	}

	__host__ __device__ void update(MaterialInfo* material, Texture*) override
	{
		const RefractiveInfo* refractive_info = (RefractiveInfo*)material;
		refractive_index_ = refractive_info->refractive_index;
	}

private:
	float refractive_index_{};
};

class Isotropic final : public Material
{
public:
	__host__ __device__ explicit Isotropic(const IsotropicInfo*, Texture* texture)
	{
		texture_ = texture;
	}

	__host__ __device__ bool scatter(Ray& ray, const Intersection& intersection, float3& absorption, uint32_t* random_state) const override
	{
		ray = Ray(intersection.point, sphere_random(random_state));
		absorption = texture_->color(intersection.uv);
		return true;
	}

	__host__ __device__ void update(MaterialInfo*, Texture* texture) override
	{
		texture_ = texture;
	}
};