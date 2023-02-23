#pragma once
#include "Intersection.cuh"
#include "Ray.cuh"
#include "../info/ObjectInfo.hpp"

#include <cfloat>

class Object
{
public:
	virtual ~Object() = default;

	__host__ __device__ virtual bool intersect(const Ray& ray, float t_min, float t_max, Intersection& intersection) const = 0;
	__host__ __device__ virtual void update(ObjectInfo* object_info, Material* material) = 0;

	Material* material_ = nullptr;
};

class Sphere final : public Object
{
public:
	__host__ __device__ Sphere(const SphereInfo* sphere_info, Material* material) : center_(sphere_info->center.str), radius_(sphere_info->radius)
	{
		material_ = material;
	}

	__host__ __device__ bool intersect(const Ray& ray, const float t_min, const float t_max, Intersection& intersection) const override
	{
		const float3 oc = ray.origin() - center_;
		const float a = dot(ray.direction(), ray.direction());
		const float b = dot(oc, ray.direction());
		const float c = dot(oc, oc) - radius_ * radius_;
		const float discriminant = b * b - a * c;

		if (discriminant < 0)
			return false;

		float t = (-b - sqrt(discriminant)) / a;
		if (t < t_max && t > t_min)
		{
			intersection.t = t;
			intersection.point = ray.position(intersection.t);
			intersection.normal = (intersection.point - center_) / radius_;
			intersection.material = material_;
			return true;
		}
		t = (-b + sqrt(discriminant)) / a;
		if (t < t_max && t > t_min)
		{
			intersection.t = t;
			intersection.point = ray.position(intersection.t);
			intersection.normal = (intersection.point - center_) / radius_;
			intersection.material = material_;
			return true;
		}
		return false;
	}

	__host__ __device__ void update(ObjectInfo* object_info, Material* material) override
	{
		const SphereInfo* sphere_info = (SphereInfo*)object_info;
		center_ = sphere_info->center.str;
		radius_ = sphere_info->radius;
		material_ = material;
	}

private:
	float3 center_{};
	float radius_{};
};

class Triangle final : public Object																																																																						
{
public:
	Triangle() = default;

	__host__ __device__ Triangle(const TriangleInfo* triangle_info, Material* material) : v0_(triangle_info->v0.str), v1_(triangle_info->v1.str), v2_(triangle_info->v2.str), normal_average_(triangle_info->normal)/*, min_uv_(triangle_info->min_uv), max_uv_(triangle_info->max_uv)*/
	{
		material_ = material;
	}

	__host__ __device__ bool intersect(const Ray& ray, const float t_min, const float t_max, Intersection& intersection) const override
	{
		const float3 v0_v1 = v1_ - v0_;
		const float3 v0_v2 = v2_ - v0_;

		const float3 p_vec = cross(ray.direction(), v0_v2);
		const float determinant = dot(p_vec, v0_v1);

		if (determinant < FLT_EPSILON)
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

		const float t = dot(q_vec, v0_v2) * inverse_determinant;

		if (t < t_min || t > t_max)
			return false;

		intersection.t = t;
		intersection.point = ray.position(intersection.t);
		float3 normal = cross(v1_ - v0_, v2_ - v0_);
		intersection.normal = dot(normal_average_, normal) < 0.0f ? -normal : normal;
		intersection.material = material_;
		return true;
	}

	__host__ __device__ void update(ObjectInfo* object_info, Material* material) override
	{
		const TriangleInfo* triangle_info = (TriangleInfo*)object_info;
		v0_ = triangle_info->v0.str;
		v1_ = triangle_info->v1.str;
		v2_ = triangle_info->v2.str;
		material_ = material;
	}

	__host__ __device__ Triangle& transform(const float3 translation, const float3 scale, const float3 rotation)
	{
		v0_ = point_transform(v0_, translation, scale, rotation);
		v1_ = point_transform(v1_, translation, scale, rotation);
		v2_ = point_transform(v2_, translation, scale, rotation);
		normal_average_ = point_transform(normal_average_, translation, scale, rotation);

		return *this;
	}

private:
	float3 v0_{}, v1_{}, v2_{};
	float3 normal_average_{};
	//float2 min_uv_, max_uv_;
};

class Plane final : public Object
{
public:
	Plane() = default;

	__host__ __device__ Plane(const PlaneInfo* plane_info, Material* material) : normal_(normalize(plane_info->normal.str)), offset_(plane_info->offset) 
	{
		material_ = material;
	}

	__host__ __device__ bool intersect(const Ray& ray, const float t_min, const float t_max, Intersection& intersection) const override
	{
		const float angle = dot(normal_, ray.direction());
		if (angle < FLT_EPSILON)
			return false;
		
		const float t = -((dot(normal_, ray.origin()) + offset_) / angle);

		if (t < t_min || t > t_max)
			return false;

		intersection.t = t;
		intersection.point = ray.position(intersection.t);
		intersection.normal = normal_;
		intersection.material = material_;
		return true;
	}

	__host__ __device__ void update(ObjectInfo* object_info, Material* material) override
	{
		const PlaneInfo* plane_info = (PlaneInfo*)object_info;
		normal_ = plane_info->normal.str;
		offset_ = plane_info->offset;
		material_ = material;
	}

private:
	float3 normal_{};
	float offset_{};
};

class Model final : public Object
{
public:
	__host__ __device__ Model(const ModelInfo* model_info, Material* material, const TriangleInfo* triangle_list) : triangle_count_(model_info->triangle_count)
	{
		material_ = material;
		triangles_ = new Triangle[triangle_count_];

		for (uint64_t i = 0; i < triangle_count_; i++)
			triangles_[i] = Triangle(&triangle_list[i], material);
	}

	~Model() override
	{
		delete[] triangles_;
	}

	__host__ __device__ bool intersect(const Ray& ray, const float t_min, const float t_max, Intersection& intersection) const override
	{
		for (uint64_t i = 0; i < triangle_count_; i++)
			if (triangles_[i].intersect(ray, t_min, t_max, intersection))
				return true;
		return false;
	}

	__host__ __device__ void update(ObjectInfo* object_info, Material* material) override
	{
		const ModelInfo* model_info = (ModelInfo*)object_info;

		material_ = material;

		for (uint64_t i = 0; i < triangle_count_; i++)
			triangles_[i] = Triangle(&model_info->triangles[i], material).transform(model_info->translation.str, model_info->scale.str, model_info->rotation.str);
	}

private:
	Triangle* triangles_ = nullptr;
	uint64_t triangle_count_{};
};