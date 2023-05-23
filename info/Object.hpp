#pragma once
#include "../common/Boundary.hpp"
#include "../common/Intersection.hpp"

#include <cstdint>

enum class ObjectType
{
	UNKNOWN_OBJECT,
	SPHERE,
	CYLINDER,
	MODEL
};

static __inline__ __host__ __device__ Boundary bound_triangle(const float3* vertices, const uint3* first_index)
{
	const uint3 index = *first_index;
	const float3 v0 = vertices[index.x];
	const float3 v1 = vertices[index.y];
	const float3 v2 = vertices[index.z];

	return { fminf(v0, v1, v2), fmaxf(v0, v1, v2) };
}

static __inline__ __host__ __device__ bool intersect_triangle(const Ray& ray, Intersection& intersection, const float3* vertices, const uint3* first_index, const float3* normals, const float2* uv)
{
	const uint3 index = *first_index;
	const float3 v0 = vertices[index.x];
	const float3 v1 = vertices[index.y];
	const float3 v2 = vertices[index.z];

	const float3 v0_v1 = v1 - v0;
	const float3 v0_v2 = v2 - v0;

	const float3 p_vec = cross(ray.direction_, v0_v2);
	const float determinant = dot(p_vec, v0_v1);

	if (determinant < FLT_EPSILON)
		return false;

	const float inverse_determinant = 1.0f / determinant;

	const float3 t_vec = ray.origin_ - v0;
	const float3 q_vec = cross(t_vec, v0_v1);

	const float u = dot(p_vec, t_vec) * inverse_determinant;
	const float v = dot(q_vec, ray.direction_) * inverse_determinant;

	if (v < 0.0f || u < 0.0f || u + v > 1.0f)
		return false;

	const float t = dot(q_vec, v0_v2) * inverse_determinant;

	if (t < kTMin || t > ray.t_max_)
		return false;

	ray.t_max_ = t;
	intersection.t = t;
	intersection.point = ray.position(intersection.t);
	intersection.uv = (1.0f - u - v) * uv[index.x] + u * uv[index.y] + v * uv[index.z];
	intersection.normal = normalize((1.0f - u - v) * normals[index.x] + u * normals[index.y] + v * normals[index.z]);
	return true;
}

struct Sphere
{
	__host__ Sphere(const float3 center, const float radius)
		: center(center), radius(radius) {}

	__inline__ __host__ __device__ bool intersect(const Ray& ray, Intersection& intersection) const
	{
		const float3 oc = ray.origin_ - center;
		const float a = dot(ray.direction_, ray.direction_);
		const float b = dot(oc, ray.direction_);
		const float c = dot(oc, oc) - radius * radius;
		const float delta = b * b - a * c;

		if (delta < 0.0f)
			return false;

		const float sqrt_delta = sqrt(delta);

		float t = (-b - sqrt_delta) / a;
		if (t < ray.t_max_ && t > kTMin)
		{
			ray.t_max_ = t;
			intersection.t = t;
			intersection.point = ray.position(intersection.t);
			intersection.normal = (intersection.point - center) / radius;

			const float u = (atan2(intersection.normal.z, intersection.normal.x) + kPi) * kInv2Pi;
			const float v = acos(intersection.normal.y) * kInvPi;

			intersection.uv = make_float2(u, v);
			return true;
		}
		t = (-b + sqrt_delta) / a;
		if (t < ray.t_max_ && t > kTMin)
		{
			ray.t_max_ = t;
			intersection.t = t;
			intersection.point = ray.position(intersection.t);
			intersection.normal = (intersection.point - center) / radius;

			const float u = (atan2(intersection.normal.z, intersection.normal.x) + kPi) * kInv2Pi;
			const float v = acos(intersection.normal.y) * kInvPi;

			intersection.uv = make_float2(u, v);
			return true;
		}
		return false;
	}

	__inline__ __host__ __device__ Boundary bound() const
	{
		return { center - make_float3(radius), center + make_float3(radius) };
	}

	float3 center{};
	float radius{};
};

struct Cylinder
{
	__host__ Cylinder(const float3 extreme_a, const float3 extreme_b, const float radius)
		: extreme_a(extreme_a), extreme_b(extreme_b), radius(radius) {}

	__inline__ __host__ __device__ bool intersect(const Ray& ray, Intersection& intersection) const
	{
		const float3 ob = ray.origin_ - extreme_b;
		const float3 axis = normalize(extreme_a - extreme_b);

		const float ba = dot(ob, axis);
		const float da = dot(ray.direction_, axis);
		const float od = dot(ray.direction_, ob);

		const float a = dot(ray.direction_, ray.direction_) - da * da;
		const float b = od - da * ba;
		const float c = dot(ob, ob) - ba * ba - radius * radius;

		const float delta = b * b - a * c;

		if (delta < 0.0f) 
			return false;

		const float sqrt_delta = sqrt(delta);

		const float t1 = (-b - sqrt_delta) / a;
		const float t2 = (-b + sqrt_delta) / a;
		const float t = t1 > t2 ? t2 : t1;

		const float m = da * t + ba;

		if (m > 0.0f && m < length(extreme_a - extreme_b))
		{
			if (t < kTMin || t > ray.t_max_)
				return false;

			ray.t_max_ = t;
			intersection.t = t;
			intersection.point = ray.position(t);
			intersection.normal = normalize(intersection.point - extreme_b - axis * m);
			intersection.uv.x = acosf(intersection.normal.x) / kPi;
			intersection.uv.y = intersection.point.y / (extreme_b.y - extreme_a.y);
			return true;
		}

		const float aa = dot(ray.origin_ - extreme_a, axis);
		const float t_top = -aa / da;
		const float3 top_point = ray.position(t_top);
		if (length(extreme_a - top_point) < radius && -da > 0.0f)
		{
			if (t_top < kTMin || t_top > ray.t_max_)
				return false;

			ray.t_max_ = t_top;
			intersection.t = t_top;
			intersection.point = top_point;
			intersection.normal = axis;
			intersection.uv = fracf(make_float2(intersection.point.x, intersection.point.z));
			return true;
		}

		const float t_bottom = -ba / da;
		const float3 bottom_point = ray.position(t_bottom);
		if (length(extreme_b - bottom_point) < radius && da > 0.0f)
		{
			if (t_bottom < kTMin || t_bottom > ray.t_max_)
				return false;

			ray.t_max_ = t_bottom;
			intersection.t = t_bottom;
			intersection.point = bottom_point;
			intersection.normal = -axis;
			intersection.uv = fracf(make_float2(intersection.point.x, intersection.point.z));
			return true;
		}

		return false;
	}

	__inline__ __host__ __device__ Boundary bound() const
	{
		const float3 a = extreme_b - extreme_a;
		const float3 e = radius * sqrt(1.0f - a * a / dot(a, a));
		return { fminf(extreme_a - e, extreme_b - e), fmaxf(extreme_a + e, extreme_b + e) };
	}

	float3 extreme_a{}, extreme_b{};
	float radius{};
};

struct Model
{
	__host__ Model(float3* h_vertices, uint3* h_indices, float3* h_normals, float2* h_uv, const uint64_t vertex_count, const uint64_t index_count)
		: h_vertices(h_vertices), h_indices(h_indices), h_normals(h_normals), h_uv(h_uv), vertex_count(vertex_count), index_count(index_count) {}

	__inline__ __host__ __device__ bool intersect(const Ray& ray, Intersection& intersection) const
	{
		Intersection temp_intersection{};
		bool intersected = false;

		for (uint64_t i = 0; i < index_count; i++)
		{
			if (intersect_triangle(ray, temp_intersection, d_vertices, d_indices + i, d_normals, d_uv))
			{
				intersected = true;
				intersection = temp_intersection;
			}
		}

		return intersected;
	}

	__inline__ __host__ __device__ Boundary bound() const
	{
		Boundary boundary{};

		for (uint64_t i = 0; i < index_count; i++)
			boundary = unite(boundary, bound_triangle(d_vertices, d_indices + i));

		return boundary;
	}

	float3* h_vertices = nullptr, * d_vertices = nullptr;
	uint3* h_indices = nullptr, * d_indices = nullptr;
	float3* h_normals = nullptr, * d_normals = nullptr;
	float2* h_uv = nullptr, * d_uv = nullptr;
	uint64_t vertex_count{}, index_count{};
};

struct Object
{
	__host__ __device__ Object() {}

	__host__ Object(Sphere&& object, const int32_t texture, const int32_t material)
		: type(ObjectType::SPHERE), texture_id(texture), material_id(material), sphere(object) {}

	__host__ Object(Cylinder&& object, const int32_t texture, const int32_t material)
		: type(ObjectType::CYLINDER), texture_id(texture), material_id(material), cylinder(object) {}

	__host__ Object(Model&& object, const int32_t texture, const int32_t material)
		: type(ObjectType::MODEL), texture_id(texture), material_id(material), model(object) {}

	__inline__ __host__ __device__ bool intersect(const Ray& ray, Intersection& intersection) const
	{
		if (type == ObjectType::SPHERE)
			return sphere.intersect(ray, intersection);
		if (type == ObjectType::CYLINDER)
			return cylinder.intersect(ray, intersection);
		if (type == ObjectType::MODEL)
			return model.intersect(ray, intersection);
		return false;
	}

	__inline__ __host__ __device__ Boundary bound() const
	{
		if (type == ObjectType::SPHERE)
			return sphere.bound();
		if (type == ObjectType::CYLINDER)
			return cylinder.bound();
		if (type == ObjectType::MODEL)
			return model.bound();
		return {};
	}

	ObjectType type{ObjectType::UNKNOWN_OBJECT};
	int32_t texture_id{0};
	int32_t material_id{0};

	union
	{
		Sphere sphere;
		Cylinder cylinder;
		Model model;
	};
};