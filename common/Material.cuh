#pragma once
#include "../info/MaterialInfo.hpp"
#include "Intersection.cuh"
#include "Ray.cuh"

class Material
{
public:
	__host__ __device__ virtual bool scatter(const Ray& ray_in, const Intersection& intersection, float3& absorption, Ray& ray_out, uint32_t* random_state) const = 0;
	virtual ~Material() = default;
};

class Diffuse final : public Material
{
public:
	__host__ __device__ explicit Diffuse(const DiffuseInfo* diffuse_info) : albedo_(diffuse_info->albedo) {}

	__host__ __device__ bool scatter(const Ray& ray_in, const Intersection& intersection, float3& absorption, Ray& ray_out, uint32_t* random_state) const override
	{
		const float3 reflected_direction = intersection.normal + sphere_random(random_state);
		ray_out = Ray(intersection.point, reflected_direction);
		absorption = albedo_;
		return true;
	}

	__host__ __device__ void update(const DiffuseInfo* diffuse_info)
	{
		albedo_ = diffuse_info->albedo;
	}

private:
	float3 albedo_;
};

class Specular final : public Material
{
public:
	__host__ __device__ explicit Specular(const SpecularInfo* specular_info) : albedo_(specular_info->albedo), fuzziness_(specular_info->fuzziness) {}

	__host__ __device__ bool scatter(const Ray& ray_in, const Intersection& intersection, float3& absorption, Ray& ray_out, uint32_t* random_state) const override
	{
		const float3 reflected_direction = reflect(versor(ray_in.direction()), intersection.normal);
		ray_out = Ray(intersection.point, reflected_direction + fuzziness_ * sphere_random(random_state));
		absorption = albedo_;
		return dot(ray_out.direction(), intersection.normal) > 0.0f;
	}

	__host__ __device__ void update(const SpecularInfo* specular_info)
	{
		albedo_ = specular_info->albedo;
		fuzziness_ = specular_info->fuzziness;
	}

private:
	float3 albedo_;
	float fuzziness_;
};

class Refractive final : public Material
{
public:
	__host__ __device__ explicit Refractive(const RefractiveInfo* refractive_info) : refractive_index_(refractive_info->refractive_index) {}

	__host__ __device__ bool scatter(const Ray& ray_in, const Intersection& intersection, float3& absorption, Ray& ray_out, uint32_t* random_state) const override
	{
		bool refracted;
		float ior;
		float reflection_probability;
		float cos_theta;
		float3 normal_out;
		float3 refracted_direction;
		const float3 reflected_direction = reflect(ray_in.direction(), intersection.normal);
		absorption = make_float3(1.0f);

		if (dot(ray_in.direction(), intersection.normal) > 0.0f)
		{
			normal_out = -intersection.normal;
			ior = refractive_index_;
			cos_theta = dot(ray_in.direction(), intersection.normal) / length(ray_in.direction());
			cos_theta = sqrt(1.0f - refractive_index_ * refractive_index_ * (1 - cos_theta * cos_theta));
		}
		else
		{
			normal_out = intersection.normal;
			ior = 1.0f / refractive_index_;
			cos_theta = -dot(ray_in.direction(), intersection.normal) / length(ray_in.direction());
		}

		const float3 unit_ray_direction = versor(ray_in.direction());
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
			ray_out = Ray(intersection.point, reflected_direction);
		}
		else
		{
			ray_out =  Ray(intersection.point, refracted_direction);
		}
		return true;
	}

	__host__ __device__ void update(const RefractiveInfo* refractive_info)
	{
		refractive_index_ = refractive_info->refractive_index;
	}

private:
	float refractive_index_;
};