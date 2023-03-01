#pragma once
#include "Intersection.cuh"
#include "Ray.cuh"
#include "../info/MaterialInfo.hpp"

class Material
{
public:
	__host__ __device__ virtual ~Material() {}

	__host__ __device__ virtual bool scatter(Ray& ray, const Intersection& intersection, float3& absorption, uint32_t* random_state) const = 0;
	__host__ __device__ virtual void update(MaterialInfo* material_info) = 0;
};

class Diffuse final : public Material
{
public:
	__host__ __device__ explicit Diffuse(const DiffuseInfo* diffuse_info) : albedo_(diffuse_info->albedo.str) {}

	__host__ __device__ bool scatter(Ray& ray, const Intersection& intersection, float3& absorption, uint32_t* random_state) const override
	{
		const float3 reflected_direction = intersection.normal + sphere_random(random_state);
		ray = Ray(intersection.point, reflected_direction);
		absorption = albedo_;
		return true;
	}

	__host__ __device__ void update(MaterialInfo* material) override
	{
		const DiffuseInfo* diffuse_info = (DiffuseInfo*)material;
		albedo_ = diffuse_info->albedo.str;
	}

private:
	float3 albedo_{};
};

class Specular final : public Material
{
public:
	__host__ __device__ explicit Specular(const SpecularInfo* specular_info) : albedo_(specular_info->albedo.str), fuzziness_(specular_info->fuzziness) {}

	__host__ __device__ bool scatter(Ray& ray, const Intersection& intersection, float3& absorption, uint32_t* random_state) const override
	{
		const float3 reflected_direction = reflect(versor(ray.direction()), intersection.normal);
		ray = Ray(intersection.point, reflected_direction + fuzziness_ * sphere_random(random_state));
		absorption = albedo_;
		return dot(ray.direction(), intersection.normal) > 0.0f;
	}

	__host__ __device__ void update(MaterialInfo* material) override
	{
		const SpecularInfo* specular_info = (SpecularInfo*)material;
		albedo_ = specular_info->albedo.str;
		fuzziness_ = specular_info->fuzziness;
	}

private:
	float3 albedo_{};
	float fuzziness_{};
};

class Refractive final : public Material
{
public:
	__host__ __device__ explicit Refractive(const RefractiveInfo* refractive_info) : refractive_index_(refractive_info->refractive_index) {}

	__host__ __device__ bool scatter(Ray& ray, const Intersection& intersection, float3& absorption, uint32_t* random_state) const override
	{
		bool refracted;
		float ior;
		float reflection_probability;
		float cos_theta;
		float3 normal_out;
		float3 refracted_direction{};
		const float3 reflected_direction = reflect(ray.direction(), intersection.normal);
		absorption = make_float3(1.0f);

		if (dot(ray.direction(), intersection.normal) > 0.0f)
		{
			normal_out = -intersection.normal;
			ior = refractive_index_;
			cos_theta = dot(ray.direction(), intersection.normal) / length(ray.direction());
			cos_theta = sqrt(1.0f - refractive_index_ * refractive_index_ * (1 - cos_theta * cos_theta));
		}
		else
		{
			normal_out = intersection.normal;
			ior = 1.0f / refractive_index_;
			cos_theta = -dot(ray.direction(), intersection.normal) / length(ray.direction());
		}

		const float3 unit_ray_direction = versor(ray.direction());
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

	__host__ __device__ void update(MaterialInfo* material) override
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
	__host__ __device__ explicit Isotropic(const IsotropicInfo* isotropic_info) : albedo_(isotropic_info->albedo.str) {}

	__host__ __device__ bool scatter(Ray& ray, const Intersection& intersection, float3& absorption, uint32_t* random_state) const override
	{
		ray = Ray(intersection.point, sphere_random(random_state));
		absorption = albedo_;
		return true;
	}

	__host__ __device__ void update(MaterialInfo* material) override
	{
		const IsotropicInfo* isotropic_info = (IsotropicInfo*)material;
		albedo_ = isotropic_info->albedo.str;
	}

private:
	float3 albedo_{};
};

class Texture final : public Material
{
public:
	__host__ __device__ explicit Texture(const TextureInfo* texture_info) : data_(texture_info->usable_data), width_(texture_info->width), height_(texture_info->height) {}

	__host__ __device__ bool scatter(Ray& ray, const Intersection& intersection, float3& absorption, uint32_t* random_state) const override
	{
		const float3 reflected_direction = intersection.normal + sphere_random(random_state);
		ray = Ray(intersection.point, reflected_direction);

		const auto i = (int32_t)(intersection.uv.x * (float)width_);
		const auto j = (int32_t)(intersection.uv.y * (float)height_);

		const int32_t texel_index = 3 * (j * width_ + i);

		if (texel_index < 0 || texel_index > 3 * width_ * height_ + 2)
			return true;

		absorption = make_float3(data_[texel_index], data_[texel_index + 1], data_[texel_index + 2]);
		return true;
	}

	__host__ __device__ void update(MaterialInfo* material) override
	{
		const TextureInfo* texture_info = (TextureInfo*)material;
		data_ = texture_info->usable_data;
		width_ = texture_info->width;
		height_ = texture_info->height;
	}

private:
	float* data_ = nullptr;
	int32_t width_{}, height_{};
};