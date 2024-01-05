// Copyright Wiktor Merta 2023
#pragma once
#include "../common/Math.hpp"

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
	__inline__ __host__ __device__ bool scatter(float3& direction, const float3& normal, uint32_t* random_state) const
	{
		const float3 reflected_direction = normal + sphere_random(random_state);
		direction = reflected_direction;
		return true;
	}
};

struct Specular
{
	__host__ Specular(const float fuzziness)
		: fuzziness(fuzziness) {}

	__inline__ __host__ __device__ bool scatter(float3& direction, const float3& normal, uint32_t* random_state) const
	{
		const float3 reflected_direction = reflect(versor(direction), normal);
		direction = reflected_direction + fuzziness * sphere_random(random_state);
		return dot(direction, normal) > 0.0f;
	}

	float fuzziness{};
};

struct Refractive
{
	__host__ Refractive(const float refractive_index)
		: refractive_index(refractive_index) {}

	__inline__ __host__ __device__ bool scatter(float3& direction, const float3& normal, uint32_t* random_state) const
	{
		bool refracted;
		float ior;
		float reflection_probability;
		float cos_theta;
		float3 normal_out;
		float3 refracted_direction{};
		const float3 reflected_direction = reflect(direction, normal);

		if (dot(direction, normal) > 0.0f)
		{
			normal_out = -normal;
			ior = refractive_index;
			cos_theta = dot(direction, normal) / length(direction);
			cos_theta = sqrt(1.0f - refractive_index * refractive_index * (1.0f - cos_theta * cos_theta));
		}
		else
		{
			normal_out = normal;
			ior = 1.0f / refractive_index;
			cos_theta = -dot(direction, normal) / length(direction);
		}

		const float3 unit_ray_direction = versor(direction);
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
			direction = reflected_direction;
		}
		else
		{
			direction = refracted_direction;
		}
		return true;
	}

	float refractive_index{};
};

struct Isotropic
{
	__inline__ __host__ __device__ bool scatter(float3& direction, uint32_t* random_state) const
	{
		direction = sphere_random(random_state);
		return true;
	}
};

struct Material
{
	__host__ __device__ Material() {}

	explicit __host__ Material(const Diffuse& material)
		: type(MaterialType::DIFFUSE), diffuse(material) {}

	explicit __host__ Material(const Specular& material)
		: type(MaterialType::SPECULAR), specular(material) {}

	explicit __host__ Material(const Refractive& material)
		: type(MaterialType::REFRACTIVE), refractive(material) {}

	explicit __host__ Material(const Isotropic& material)
		: type(MaterialType::ISOTROPIC), isotropic(material) {}

	// Calculating new direction based on current material
	__inline__ __host__ __device__ bool scatter(float3& direction, const float3& normal, uint32_t* random_state) const
	{
		if (type == MaterialType::DIFFUSE)
			return diffuse.scatter(direction, normal, random_state);
		if (type == MaterialType::SPECULAR)
			return specular.scatter(direction, normal, random_state);
		if (type == MaterialType::REFRACTIVE)
			return refractive.scatter(direction, normal, random_state);
		if (type == MaterialType::ISOTROPIC)
			return isotropic.scatter(direction, random_state);
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