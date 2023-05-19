#pragma once
#include "../common/Ray.hpp"
#include "../common/Intersection.hpp"

#include <cstdint>

enum class MaterialType
{
	UNKNOWN_MATERIAL,
	DIFFUSE,
	SPECULAR,
	REFRACTIVE,
	ISOTROPIC
};

struct Diffuse
{
	__inline__ __host__ __device__ bool scatter(Ray& ray, const Intersection& intersection, uint32_t* random_state) const
	{
		const float3 reflected_direction = intersection.normal + sphere_random(random_state);
		ray = Ray(intersection.point, reflected_direction);
		return true;
	}
};

struct Specular
{
	__host__ Specular(const float fuzziness)
		: fuzziness(fuzziness) {}

	__inline__ __host__ __device__ bool scatter(Ray& ray, const Intersection& intersection, uint32_t* random_state) const
	{
		const float3 reflected_direction = reflect(versor(ray.direction_), intersection.normal);
		ray = Ray(intersection.point, reflected_direction + fuzziness * sphere_random(random_state));
		return dot(ray.direction_, intersection.normal) > 0.0f;
	}

	float fuzziness{};
};

struct Refractive
{
	__host__ Refractive(const float refractive_index)
		: refractive_index(refractive_index) {}

	__inline__ __host__ __device__ bool scatter(Ray& ray, const Intersection& intersection, uint32_t* random_state) const
	{
		bool refracted;
		float ior;
		float reflection_probability;
		float cos_theta;
		float3 normal_out;
		float3 refracted_direction{};
		const float3 reflected_direction = reflect(ray.direction_, intersection.normal);

		if (dot(ray.direction_, intersection.normal) > 0.0f)
		{
			normal_out = -intersection.normal;
			ior = refractive_index;
			cos_theta = dot(ray.direction_, intersection.normal) / length(ray.direction_);
			cos_theta = sqrt(1.0f - refractive_index * refractive_index * (1.0f - cos_theta * cos_theta));
		}
		else
		{
			normal_out = intersection.normal;
			ior = 1.0f / refractive_index;
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
			float r0 = (1.0f - refractive_index) / (1.0f + refractive_index);
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
			ray = Ray(intersection.point, refracted_direction);
		}
		return true;
	}

	float refractive_index{};
};

struct Isotropic
{
	__inline__ __host__ __device__ bool scatter(Ray& ray, const Intersection& intersection, uint32_t* random_state) const
	{
		ray = Ray(intersection.point, sphere_random(random_state));
		return true;
	}
};

struct Material
{
	__host__ __device__ Material() {}

	explicit __host__ Material(Diffuse&& material)
		: type(MaterialType::DIFFUSE), diffuse(material) {}

	explicit __host__ Material(Specular&& material)
		: type(MaterialType::SPECULAR), specular(material) {}

	explicit __host__ Material(Refractive&& material)
		: type(MaterialType::REFRACTIVE), refractive(material) {}

	explicit __host__ Material(Isotropic&& material)
		: type(MaterialType::ISOTROPIC), isotropic(material) {}

	__inline__ __host__ __device__ bool scatter(Ray& ray, const Intersection& intersection, uint32_t* random_state) const
	{
		if (type == MaterialType::DIFFUSE)
			return diffuse.scatter(ray, intersection, random_state);
		if (type == MaterialType::SPECULAR)
			return specular.scatter(ray, intersection, random_state);
		if (type == MaterialType::REFRACTIVE)
			return refractive.scatter(ray, intersection, random_state);
		if (type == MaterialType::ISOTROPIC)
			return isotropic.scatter(ray, intersection, random_state);
		return false;
	}

	MaterialType type{MaterialType::UNKNOWN_MATERIAL};

	union
	{
		Diffuse diffuse;
		Specular specular;
		Refractive refractive;
		Isotropic isotropic;
	};
};