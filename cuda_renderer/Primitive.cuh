#pragma once
#include "Intersection.cuh"

class Primitive
{
public:
	__device__ virtual bool intersect(const Ray& ray, float t_min, float t_max, Intersection& intersection) const = 0;
	virtual ~Primitive() = default;
};

class Sphere final : public Primitive
{
public:
	__device__ Sphere(const float3 center, const float radius, Material* material) : material_(material), center_(center), radius_(radius) {}
	__device__ bool intersect(const Ray& ray, const float t_min, const float t_max, Intersection& intersection) const override
	{
		const float3 oc = ray.origin() - center_;
		const float a = dot(ray.direction(), ray.direction());
		const float b = dot(oc, ray.direction());
		const float c = dot(oc, oc) - radius_ * radius_;
		const float discriminant = b * b - a * c;

		if (discriminant > 0)
		{
			float temp = (-b - sqrt(discriminant)) / a;
			if (temp < t_max && temp > t_min)
			{
				intersection.t = temp;
				intersection.point = ray.position(intersection.t);
				intersection.normal = (intersection.point - center_) / radius_;
				intersection.material = material_;
				return true;
			}
			temp = (-b + sqrt(discriminant)) / a;
			if (temp < t_max && temp > t_min)
			{
				intersection.t = temp;
				intersection.point = ray.position(intersection.t);
				intersection.normal = (intersection.point - center_) / radius_;
				intersection.material = material_;
				return true;
			}
		}
		return false;
	}

	Material* material_;

private:
	float3 center_;
	float radius_;
};