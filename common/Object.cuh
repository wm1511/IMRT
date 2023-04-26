#pragma once
#include "Intersection.cuh"
#include "Boundary.cuh"
#include "../info/ObjectInfo.hpp"

#include <cfloat>


class Object
{
public:
	__host__ __device__ virtual ~Object() {}

	__host__ __device__ virtual bool intersect(const Ray& ray, Intersection& intersection) const = 0;
	__host__ __device__ virtual Boundary bound() = 0;
	__host__ __device__ virtual void update(ObjectInfo* object_info, Material* material) = 0;

	Material* material_ = nullptr;
};

class Sphere final : public Object
{
public:
	Sphere() = default;

	__host__ __device__ Sphere(const SphereInfo* sphere_info, Material* material)
		: center_(sphere_info->center.str), radius_(sphere_info->radius)
	{
		material_ = material;
	}

	__host__ __device__ bool intersect(const Ray& ray, Intersection& intersection) const override
	{
		const float3 oc = ray.origin_ - center_;
		const float a = dot(ray.direction_, ray.direction_);
		const float b = dot(oc, ray.direction_);
		const float c = dot(oc, oc) - radius_ * radius_;
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
			intersection.normal = (intersection.point - center_) / radius_;

			const float u = (atan2(intersection.normal.z, intersection.normal.x) + kPi) * kInv2Pi;
			const float v = acos(intersection.normal.y) * kInvPi;

			intersection.uv = make_float2(u, v);
			intersection.material = material_;
			return true;
		}
		t = (-b + sqrt_delta) / a;
		if (t < ray.t_max_ && t > kTMin)
		{
			ray.t_max_ = t;
			intersection.t = t;
			intersection.point = ray.position(intersection.t);
			intersection.normal = (intersection.point - center_) / radius_;

			const float u = (atan2(intersection.normal.z, intersection.normal.x) + kPi) * kInv2Pi;
			const float v = acos(intersection.normal.y) * kInvPi;

			intersection.uv = make_float2(u, v);
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

	__host__ __device__ Boundary bound() override
	{
		return { center_ - make_float3(radius_), center_ + make_float3(radius_) };
	}

private:
	float3 center_{};
	float radius_{};
};

class Triangle final : public Object
{
public:
	Triangle() = default;

	__host__ __device__ Triangle(const TriangleInfo* triangle_info, Material* material)
		: v0_(triangle_info->v0.str), v1_(triangle_info->v1.str), v2_(triangle_info->v2.str), normal_average_(triangle_info->normal), min_uv_(triangle_info->min_uv), max_uv_(triangle_info->max_uv)
	{
		material_ = material;
	}

	__host__ __device__ bool intersect(const Ray& ray, Intersection& intersection) const override
	{
		const float3 v0_v1 = v1_ - v0_;
		const float3 v0_v2 = v2_ - v0_;

		const float3 p_vec = cross(ray.direction_, v0_v2);
		const float determinant = dot(p_vec, v0_v1);

		if (determinant < FLT_EPSILON)
			return false;

		const float inverse_determinant = 1.0f / determinant;

		const float3 t_vec = ray.origin_ - v0_;
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
		intersection.normal = normal_average_;
		intersection.uv = make_float2(lerp(min_uv_.x, max_uv_.x, u), lerp(min_uv_.y, max_uv_.y, v));
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
		transform_point(v0_, translation, scale, rotation);
		transform_point(v1_, translation, scale, rotation);
		transform_point(v2_, translation, scale, rotation);

		rotate_point_x(normal_average_, rotation.x);
		rotate_point_y(normal_average_, rotation.y);
		rotate_point_z(normal_average_, rotation.z);

		return *this;
	}

	__host__ __device__ Boundary bound() override
	{
		return { fminf(v0_, v1_, v2_), fmaxf(v0_, v1_, v2_) };
	}

private:
	float3 v0_{}, v1_{}, v2_{};
	float3 normal_average_{};
	float2 min_uv_{}, max_uv_{};
};

class Plane final : public Object
{
public:
	Plane() = default;

	__host__ __device__ Plane(const PlaneInfo* plane_info, Material* material)
		: normal_(normalize(plane_info->normal.str)), offset_(plane_info->offset)
	{
		material_ = material;
	}

	__host__ __device__ bool intersect(const Ray& ray, Intersection& intersection) const override
	{
		const float angle = dot(normal_, ray.direction_);
		if (angle < FLT_EPSILON)
			return false;

		const float t = -((dot(normal_, ray.origin_) + offset_) / angle);

		if (t < kTMin || t > ray.t_max_)
			return false;

		ray.t_max_ = t;
		intersection.t = t;
		intersection.point = ray.position(intersection.t);
		intersection.normal = normal_;
		intersection.uv = fracf(make_float2(intersection.point.x, intersection.point.z));
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

	__host__ __device__ Boundary bound() override
	{
		return { make_float3(-FLT_MAX), make_float3(FLT_MAX) };
	}

private:
	float3 normal_{};
	float offset_{};
};

class Cylinder final : public Object
{
public:
	Cylinder() = default;

	__host__ __device__ Cylinder(const CylinderInfo* cylinder_info, Material* material)
		: extreme_a_(cylinder_info->extreme_a.str), extreme_b_(cylinder_info->extreme_b.str), radius_(cylinder_info->radius)
	{
		material_ = material;
	}

	__host__ __device__ bool intersect(const Ray& ray, Intersection& intersection) const override
	{
		const float3 ob = ray.origin_ - extreme_b_;
		const float3 axis = normalize(extreme_a_ - extreme_b_);

		const float ba = dot(ob, axis);
		const float da = dot(ray.direction_, axis);
		const float od = dot(ray.direction_, ob);

		const float a = dot(ray.direction_, ray.direction_) - da * da;
		const float b = od - da * ba;
		const float c = dot(ob, ob) - ba * ba - radius_ * radius_;

		const float delta = b * b - a * c;

		if (delta < 0.0f) 
			return false;

		const float sqrt_delta = sqrt(delta);

		const float t1 = (-b - sqrt_delta) / a;
		const float t2 = (-b + sqrt_delta) / a;
		const float t = t1 > t2 ? t2 : t1;

		const float m = da * t + ba;

		if (m > 0.0f && m < length(extreme_a_ - extreme_b_))
		{
			if (t < kTMin || t > ray.t_max_)
				return false;

			ray.t_max_ = t;
			intersection.t = t;
			intersection.point = ray.position(t);
			intersection.normal = normalize(intersection.point - extreme_b_ - axis * m);
			intersection.uv.x = acosf(intersection.normal.x) / kPi;
			intersection.uv.y = intersection.point.y / (extreme_b_.y - extreme_a_.y);
			intersection.material = material_;
			return true;
		}

		const float aa = dot(ray.origin_ - extreme_a_, axis);
		const float t_top = -aa / da;
		const float3 top_point = ray.position(t_top);
		if (length(extreme_a_ - top_point) < radius_ && -da > 0.0f)
		{
			if (t_top < kTMin || t_top > ray.t_max_)
				return false;

			ray.t_max_ = t_top;
			intersection.t = t_top;
			intersection.point = top_point;
			intersection.normal = axis;
			intersection.uv = fracf(make_float2(intersection.point.x, intersection.point.z));
			intersection.material = material_;
			return true;
		}

		const float t_bottom = -ba / da;
		const float3 bottom_point = ray.position(t_bottom);
		if (length(extreme_b_ - bottom_point) < radius_ && da > 0.0f)
		{
			if (t_bottom < kTMin || t_bottom > ray.t_max_)
				return false;

			ray.t_max_ = t_bottom;
			intersection.t = t_bottom;
			intersection.point = bottom_point;
			intersection.normal = -axis;
			intersection.uv = fracf(make_float2(intersection.point.x, intersection.point.z));
			intersection.material = material_;
			return true;
		}

		return false;
	}

	__host__ __device__ void update(ObjectInfo* object_info, Material* material) override
	{
		const CylinderInfo* cylinder_info = (CylinderInfo*)object_info;
		extreme_a_ = cylinder_info->extreme_a.str;
		extreme_b_ = cylinder_info->extreme_b.str;
		radius_ = cylinder_info->radius;
		material_ = material;
	}

	__host__ __device__ Boundary bound() override
	{
		const float3 a = extreme_b_ - extreme_a_;
		const float3 e = radius_ * sqrt(1.0f - a * a / dot(a, a));
		return { fminf(extreme_a_ - e, extreme_b_ - e), fmaxf(extreme_a_ + e, extreme_b_ + e) };
	}

private:
	float3 extreme_a_{}, extreme_b_{};
	float radius_{};
};

class Cone final : public Object
{
public:
	Cone() = default;

	__host__ __device__ Cone(const ConeInfo* cone_info, Material* material)
		: extreme_a_(cone_info->extreme_a.str), extreme_b_(cone_info->extreme_b.str), radius_(cone_info->radius)
	{
		material_ = material;
	}

	__host__ __device__ bool intersect(const Ray& ray, Intersection& intersection) const override
	{
		const float3 oa = ray.origin_ - extreme_a_;
		const float3 axis = normalize(extreme_b_ - extreme_a_);

		const float aa = dot(oa, axis);
		const float da = dot(ray.direction_, axis);
		const float od = dot(ray.direction_, oa);

		const float k = radius_ * radius_ / dot(extreme_b_ - extreme_a_, extreme_b_ - extreme_a_);

		const float a = dot(ray.direction_, ray.direction_) - (1.0f + k) * da * da;
		const float b = od - (1.0f + k) * da * aa;
		const float c = dot(oa, oa) - (1.0f + k) * aa * aa;

		const float delta = b * b - a * c;

		if (delta < 0.0f) 
			return false;

		const float sqrt_delta = sqrt(delta);

		const float t1 = (-b - sqrt_delta) / a;
		const float t2 = (-b + sqrt_delta) / a;
		const float t = t1 > t2 ? t2 : t1;

		const float m = da * t + aa;

		if (m > 0.0f && m < length(extreme_b_ - extreme_a_))
		{
			if (t < kTMin || t > ray.t_max_)
				return false;

			ray.t_max_ = t;
			intersection.t = t;
			intersection.point = ray.position(t);
			intersection.normal = normalize(intersection.point - extreme_a_ - (1.0f + k) * axis * m);
			intersection.uv.x = acosf(intersection.normal.x) / kPi;
			intersection.uv.y = intersection.point.y / (extreme_b_.y - extreme_a_.y);
			intersection.material = material_;
			return true;
		}

		const float ba = dot(ray.origin_ - extreme_b_, axis);
		const float t_bottom = -ba / da;
		const float3 bottom_point = ray.position(t_bottom);
		if (length(extreme_b_ - bottom_point) < radius_ && -da > 0.0f)
		{
			if (t_bottom < kTMin || t_bottom > ray.t_max_)
				return false;

			ray.t_max_ = t_bottom;
			intersection.t = t_bottom;
			intersection.point = bottom_point;
			intersection.normal = axis;
			intersection.uv = fracf(make_float2(intersection.point.x, intersection.point.z));
			intersection.material = material_;
			return true;
		}

		return false;
	}

	__host__ __device__ void update(ObjectInfo* object_info, Material* material) override
	{
		const ConeInfo* cone_info = (ConeInfo*)object_info;
		extreme_a_ = cone_info->extreme_a.str;
		extreme_b_ = cone_info->extreme_b.str;
		radius_ = cone_info->radius;
		material_ = material;
	}

	__host__ __device__ Boundary bound() override
	{
		const float3 a = extreme_b_ - extreme_a_;
		const float3 e = sqrt(1.0f - a * a / dot(a, a));
		return { fminf(extreme_a_ - e * radius_, extreme_b_ - e * radius_), fmaxf(extreme_a_ + e * radius_, extreme_b_ + e * radius_) };
	}

private:
	float3 extreme_a_{}, extreme_b_{};
	float radius_{};
};

class Model final : public Object
{
public:
	__host__ __device__ Model(const ModelInfo* model_info, Material* material)
		: triangle_count_(model_info->triangle_count)
	{
		material_ = material;
		triangles_ = (Triangle*)malloc(sizeof(Triangle) * triangle_count_);

		for (uint64_t i = 0; i < triangle_count_; i++)
			triangles_[i] = Triangle(&model_info->usable_triangles[i], material);
	}

	__host__ __device__ ~Model() override
	{
		free(triangles_);
	}

	__host__ __device__ bool intersect(const Ray& ray, Intersection& intersection) const override
	{
		Intersection temp_intersection{};
		bool intersected = false;

		for (uint64_t i = 0; i < triangle_count_; i++)
		{
			if (triangles_[i].intersect(ray, temp_intersection))
			{
				intersected = true;
				intersection = temp_intersection;
			}
		}

		return intersected;
	}

	__host__ __device__ void update(ObjectInfo* object_info, Material* material) override
	{
		const ModelInfo* model_info = (ModelInfo*)object_info;

		material_ = material;

		for (uint64_t i = 0; i < triangle_count_; i++)
			triangles_[i] = Triangle(&model_info->usable_triangles[i], material).transform(model_info->translation.str, model_info->scale.str, model_info->rotation.str);
	}

	__host__ __device__ Boundary bound() override
	{
		Boundary boundary{};

		for (uint64_t i = 0; i < triangle_count_; i++)
			boundary = unite(boundary, triangles_[i].bound());

		return boundary;
	}

private:
	Triangle* triangles_ = nullptr;
	uint64_t triangle_count_{};
};