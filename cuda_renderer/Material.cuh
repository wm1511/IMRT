#pragma once
#include "Ray.cuh"
#include "World.cuh"

#include <curand_kernel.h>

struct Intersection;

__device__ inline float3 sphere_random(curandState* random_state)
{
	float3 v;
	do
	{
		v = make_float3(curand_uniform(random_state), curand_uniform(random_state), curand_uniform(random_state)) - make_float3(1.0f);
	} while (dot(v, v) >= 1.0f);
	return v;
}

class Material
{
public:
	__device__ virtual bool scatter(const Ray& ray_in, const Intersection& intersection, float3& absorption, Ray& ray_out, curandState* random_state) const = 0;
	virtual ~Material() = default;
};

class Diffuse final : public Material
{
public:
	__device__ explicit Diffuse(const float3& albedo) : albedo_(albedo) {}
	__device__ bool scatter(const Ray& ray_in, const Intersection& intersection, float3& absorption, Ray& ray_out, curandState* random_state) const override
	{
		const float3 reflected_direction = intersection.point + intersection.normal + sphere_random(random_state);
		ray_out = Ray(intersection.point, reflected_direction - intersection.point);
		absorption = albedo_;
		return true;
	}

private:
	float3 albedo_;
};

class Specular final : public Material
{
public:
	__device__ Specular(const float3& albedo, const float fuzziness) : albedo_(albedo), fuzziness_(fuzziness) {}
	__device__ bool scatter(const Ray& ray_in, const Intersection& intersection, float3& absorption, Ray& ray_out, curandState* random_state) const override
	{
		const float3 reflected_direction = reflect(versor(ray_in.direction()), intersection.normal);
		ray_out = Ray(intersection.point, reflected_direction + fuzziness_ * sphere_random(random_state));
		absorption = albedo_;
		return dot(ray_out.direction(), intersection.normal) > 0.0f;
	}

private:
	float3 albedo_;
	float fuzziness_;
};

class Refractive final : public Material
{
public:
	__device__ explicit Refractive(const float refraction_index) : refraction_index_(refraction_index) {}
	__device__ bool scatter(const Ray& ray_in, const Intersection& intersection, float3& absorption, Ray& ray_out, curandState* random_state) const override
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
			normal_out = -1.0f * intersection.normal;
			ior = refraction_index_;
			cos_theta = dot(ray_in.direction(), intersection.normal) / length(ray_in.direction());
			cos_theta = sqrt(1.0f - refraction_index_ * refraction_index_ * (1 - cos_theta * cos_theta));
		}
		else
		{
			normal_out = intersection.normal;
			ior = 1.0f / refraction_index_;
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
			float r0 = (1.0f - refraction_index_) / (1.0f + refraction_index_);
			r0 = r0 * r0;
			reflection_probability = r0 + (1.0f - r0) * pow((1.0f - cos_theta), 5.0f);
		}
		else
		{
			reflection_probability = 1.0f;
		}

		if (curand_uniform(random_state) < reflection_probability)
		{
			ray_out = Ray(intersection.point, reflected_direction);
		}
		else
		{
			ray_out =  Ray(intersection.point, refracted_direction);
		}
		return true;
	}

private:
	float refraction_index_;
};