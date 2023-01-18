#pragma once
#include "Intersection.cuh"
#include "Ray.cuh"

class Primitive
{
public:
	__device__ virtual bool intersect(const Ray& ray, float t_min, float t_max, Intersection& intersection) const = 0;
	virtual ~Primitive() = default;

	Material* material_ = nullptr;
};

class Sphere final : public Primitive
{
public:
	__device__ Sphere(const float3 center, const float radius, Material* material) : center_(center), radius_(radius) { material_ = material; }
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
	
private:
	float3 center_;
	float radius_;
};

class Triangle final : public Primitive																																																																						
{
public:
	__device__ Triangle(const float3 v0, const float3 v1, const float3 v2, Material* material) : v0_(v0), v1_(v1), v2_(v2) { material_ = material; }
	__device__ bool intersect(const Ray& ray, const float t_min, const float t_max, Intersection& intersection) const override
	{
		const float3 v0_v1 = v1_ - v0_;
		const float3 v0_v2 = v2_ - v0_;

		const float3 p_vec = cross(ray.direction(), v0_v2);

		const float determinant = dot(p_vec, v0_v1);

		if (determinant < t_min || determinant > t_max)
			return false;

		const float inverse_determinant = 1.0f / determinant;

		const float3 t_vec = ray.origin() - v0_;
		const float u = dot(p_vec, t_vec) * inverse_determinant;
		if (u < 0 || u > 1)
			return false;

		const float3 q_vec = cross(t_vec, v0_v1);
		const float v = dot(q_vec, ray.direction()) * inverse_determinant;
		if (v < 0 || u + v > 1)
			return false;

		intersection.t = dot(q_vec, v0_v2) * inverse_determinant;
		intersection.point = ray.position(intersection.t);
		intersection.normal = cross(v1_ - v0_, v2_ - v0_);
		intersection.material = material_;
		return true;
	}

private:
	float3 v0_, v1_, v2_;
};