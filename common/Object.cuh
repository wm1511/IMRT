#pragma once
#include "Intersection.cuh"
#include "Transform.cuh"
#include "../info/ObjectInfo.hpp"

#include <cfloat>

class Object
{
public:
	__host__ __device__ virtual ~Object() {}

	__host__ __device__ virtual bool intersect(const Ray& ray, Intersection& intersection, uint32_t* random_state) const = 0;
	__host__ __device__ virtual Boundary bound() = 0;
	__host__ __device__ virtual void update(ObjectInfo* object_info, Material* material) = 0;

	Material* material_ = nullptr;
	Transform* world_to_object_ = nullptr, * object_to_world_ = nullptr;
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

	__host__ __device__ bool intersect(const Ray& ray, Intersection& intersection, uint32_t*) const override
	{
		const float3 oc = ray.origin_ - center_;
		const float a = dot(ray.direction_, ray.direction_);
		const float b = dot(oc, ray.direction_);
		const float c = dot(oc, oc) - radius_ * radius_;
		const float discriminant = b * b - a * c;

		if (discriminant < 0)
			return false;

		float t = (-b - sqrt(discriminant)) / a;
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
		t = (-b + sqrt(discriminant)) / a;
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
		return {center_ - make_float3(radius_), center_ + make_float3(radius_)};
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

	__host__ __device__ bool intersect(const Ray& ray, Intersection& intersection, uint32_t*) const override
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
		return {fminf(v0_, v1_, v2_), fmaxf(v0_, v1_, v2_)};
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

	__host__ __device__ bool intersect(const Ray& ray, Intersection& intersection, uint32_t*) const override
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
		return {make_float3(-FLT_MAX), make_float3(FLT_MAX)};
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
		: extreme_a_(cylinder_info->extreme_a.str), extreme_b_(cylinder_info->extreme_b.str), center_(cylinder_info->center.str), radius_(cylinder_info->radius)
	{
		material_ = material;		
	}

	__host__ __device__ bool intersect(const Ray& ray, Intersection& intersection, uint32_t*) const override
	{
		float3 ba = extreme_b_ - extreme_a_;
		const float3 oc = center_ + ray.origin_ - extreme_a_;

		const float baba = dot(ba, ba);
		const float bard = dot(ba, ray.direction_);
		const float baoc = dot(ba, oc);

		const float k2 = baba - bard * bard;
		const float k1 = baba * dot(oc, ray.direction_) - baoc * bard;
		const float k0 = baba * dot(oc, oc) - baoc * baoc - radius_ * radius_ * baba;
	    float h = k1 * k1 - k2 * k0;

	    if (h < FLT_EPSILON) 
			return false;

	    h = sqrt(h);

	    float t = (-k1 - h) / k2;

		if (t < kTMin || t > ray.t_max_)
			return false;

		const float y = baoc + t * bard;
	    if (y > 0.0f && y < baba )
	    {
			ray.t_max_ = t;
			intersection.t = t;
			intersection.point = ray.position(t);
			intersection.normal = (oc + t * ray.direction_ - ba * y / baba) / radius_;
			intersection.material = material_;
			return true;
		}

	    t = ((y < 0.0f ? 0.0f : baba) - baoc) / bard;

		if (t < kTMin || t > ray.t_max_)
			return false;

	    if (abs(k1 + k2 * t) < h)
	    {
			ray.t_max_ = t;
			intersection.t = t;
			intersection.point = ray.position(t);
			intersection.normal = y < 0.0f ? -ba / sqrt(baba) : ba / sqrt(baba);
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
		center_ = cylinder_info->center.str;
		radius_ = cylinder_info->radius;
		material_ = material;
	}

	__host__ __device__ Boundary bound() override
	{
		const float3 a = extreme_b_ - extreme_a_;
		const float3 e = radius_ * sqrt(1.0f - a * a / dot(a, a));
		return {fminf(extreme_a_ - e, extreme_b_ - e), fmaxf(extreme_a_ + e, extreme_b_ + e)};
	}

private:
	float3 extreme_a_{}, extreme_b_{}, center_{};
	float radius_{};
};

class Cone final : public Object
{
public:
	Cone() = default;

	__host__ __device__ Cone(const ConeInfo* cone_info, Material* material)
		: extreme_a_(cone_info->extreme_a.str), extreme_b_(cone_info->extreme_b.str), center_(cone_info->center.str), radius_a_(cone_info->radius_a), radius_b_(cone_info->radius_a)
	{
		material_ = material;		
	}

	__host__ __device__ bool intersect(const Ray& ray, Intersection& intersection, uint32_t*) const override
	{
		float3 ba = extreme_b_ - extreme_a_;
		const float3 oa = center_ + ray.origin_ - extreme_a_;
		const float3 ob = center_ + ray.origin_ - extreme_b_;
		const float m0 = dot(ba, ba);
		const float m1 = dot(oa, ba);
		const float m2 = dot(ray.direction_, ba);
		const float m3 = dot(ray.direction_, oa);
		const float m5 = dot(oa, oa);
		const float m9 = dot(ob, ba); 

	    if (m1 < 0.0f)
	    {
		    const float3 c = oa * m2 - ray.direction_ * m1;
	        if (dot(c, c) < radius_a_ * radius_a_ * m2 * m2)
	        {
				intersection.t = -m1 / m2;
				intersection.point = ray.position(intersection.t);
				intersection.normal = -ba * rsqrtf(m0);
				intersection.material = material_;
				return true;
			}
	    }
	    else if (m9 > 0.0f)
	    {
		    const float t = -m9/m2;

			if (t < kTMin || t > ray.t_max_)
				return false;

		    const float3 c = ob + ray.direction_ * t;
	        if (dot(c, c) < radius_b_ * radius_b_)
	        {
				ray.t_max_ = t;
		        intersection.t = t;
				intersection.point = ray.position(intersection.t);
				intersection.normal = ba * rsqrtf(m0);
				intersection.material = material_;
				return true;
	        }
	    }

		const float rr = radius_a_ - radius_b_;
		const float hy = m0 + rr * rr;
		const float k2 = m0 * m0 - m2 * m2 * hy;
		const float k1 = m0 * m0 * m3 - m1 * m2 * hy + m0 * radius_a_ * (rr * m2 * 1.0f);
		const float k0 = m0 * m0 * m5 - m1 * m1 * hy + m0 * radius_a_ * (rr * m1 * 2.0f - m0 * radius_a_);
		const float h = k1*k1 - k2*k0;

	    if (h < FLT_EPSILON)
			return false;

		const float t = (-k1 - sqrt(h)) / k2;

		if (t < kTMin || t > ray.t_max_)
			return false;

		const float y = m1 + t * m2;

	    if(y < 0.0f || y > m0)
			return false;

		ray.t_max_ = t;
		intersection.t = t;
		intersection.point = ray.position(intersection.t);
		intersection.normal = normalize(m0 * (m0 * (oa + t * ray.direction_) + rr * ba * radius_a_) - ba * hy * y);
		intersection.material = material_;
		return true;
	}

	__host__ __device__ void update(ObjectInfo* object_info, Material* material) override
	{
		const ConeInfo* cone_info = (ConeInfo*)object_info;
		extreme_a_ = cone_info->extreme_a.str;
		extreme_b_ = cone_info->extreme_b.str;
		center_ = cone_info->center.str;
		radius_a_ = cone_info->radius_a;
		radius_b_ = cone_info->radius_b;
		material_ = material;
	}

	__host__ __device__ Boundary bound() override
	{
		const float3 a = extreme_b_ - extreme_a_;
		const float3 e = sqrt(1.0f - a * a / dot(a, a));
		return {fminf(extreme_a_ - e * radius_a_, extreme_b_ - e * radius_b_), fmaxf(extreme_a_ + e * radius_a_, extreme_b_ + e * radius_b_)};
	}

private:
	float3 extreme_a_{}, extreme_b_{}, center_{};
	float radius_a_{}, radius_b_{};
};

class Torus final : public Object
{
public:
	Torus() = default;

	__host__ __device__ Torus(const TorusInfo* torus_info, Material* material)
		: center_(torus_info->center.str) , radius_a_(torus_info->radius_a), radius_b_(torus_info->radius_a)
	{
		material_ = material;		
	}

	__host__ __device__ bool intersect(const Ray& ray, Intersection& intersection, uint32_t*) const override
	{
		float po = 1.0f;
		const float Ra2 = radius_a_ * radius_a_;
		const float ra2 = radius_b_ * radius_b_;
		const float m = dot(center_ + ray.origin_, center_ + ray.origin_);
		const float n = dot(center_ + ray.origin_, ray.direction_);
		const float k = (m + Ra2 - ra2) / 2.0f;
	    float k3 = n;
		const float2 dxy = make_float2(ray.direction_.x, ray.direction_.y);
		const float2 oxy = make_float2(center_.x + ray.origin_.x, center_.y + ray.origin_.y);
	    float k2 = n * n - Ra2 * dot(dxy, dxy) + k;
	    float k1 = n * k - Ra2 * dot(dxy, oxy);
	    float k0 = k * k - Ra2 * dot(oxy, oxy);
	    
	    if (abs(k3 * (k3 * k3 - k2) + k1) < 0.01f)
	    {
	        po = -1.0f;
	        const float temp = k1;
	    	k1 = k3;
	    	k3 = temp;
	        k0 = 1.0f / k0;
	        k1 = k1 * k0;
	        k2 = k2 * k0;
	        k3 = k3 * k0;
	    }
	    
	    float c2 = k2 * 2.0f - 3.0f * k3 * k3;
	    float c1 = k3 * (k3 * k3 - k2) + k1;
	    float c0 = k3 * (k3 * (c2 + 2.0f * k2) - 8.0f * k1) + 4.0f * k0;
	    c2 /= 3.0f;
	    c1 *= 2.0f;
	    c0 /= 3.0f;
		const float Q = c2 * c2 + c0;
		const float R = c2 * c2 * c2 - 3.0f * c2 * c0 + c1 * c1;
	    float h = R * R - Q * Q * Q;
	    
	    if (h >= FLT_EPSILON)
	    {
	        h = sqrt(h);
	        const float v = R + h > 0.0f ? pow(abs(R + h), 1.0f / 3.0f) : -pow(abs(R + h), 1.0f / 3.0f);
	        const float u = R - h > 0.0f ? pow(abs(R - h), 1.0f / 3.0f) : -pow(abs(R - h), 1.0f / 3.0f);
	        const float2 s = make_float2(v + u + 4.0f * c2, (v - u) * sqrt(3.0f));
	        const float y = sqrt(0.5f * (length(s) + s.x));
	        const float x = 0.5f * s.y / y;
	        const float r = 2.0f * c1 / (x * x + y * y);
	        float t1 = x - r - k3;
	    	t1 = po < 0.0f ? 2.0f / t1 : t1;
	        float t2 = -x - r - k3;
	    	t2 = po < 0.0f ? 2.0f / t2 : t2;
	        float t = 1e20f;

	        if (t1 > 0.0f) 
				t = t1;
	        if (t2 > 0.0f)
				t = fmin(t, t2);

			if (t < kTMin || t > ray.t_max_)
				return false;

			ray.t_max_ = t;
			intersection.t = t;
			intersection.point = ray.position(intersection.t);
			intersection.normal = normalize(intersection.point * (dot(intersection.point, intersection.point) - radius_b_ * radius_b_ - radius_b_ * radius_b_ * make_float3(1.0f, 1.0f, -1.0f)));
			intersection.uv = make_float2(u, v);
			intersection.material = material_;
			return true;
	    }

		const float sQ = sqrt(Q);
		const float w = sQ * cos(acos(-R / (sQ * Q)) / 3.0f);
		const float d2 = -(w + c2);

		if (d2 < 0.0f)
			return false;

		const float d1 = sqrt(d2);
		const float h1 = sqrt(w - 2.0f * c2 + c1 / d1);
		const float h2 = sqrt(w - 2.0f * c2 - c1 / d1);
	    float t1 = -d1 - h1 - k3; t1 = po < 0.0f ? 2.0f / t1 : t1;
	    float t2 = -d1 + h1 - k3; t2 = po < 0.0f ? 2.0f / t2 : t2;
	    float t3 =  d1 - h2 - k3; t3 = po < 0.0f ? 2.0f / t3 : t3;
	    float t4 =  d1 + h2 - k3; t4 = po < 0.0f ? 2.0f / t4 : t4;
	    float t = 1e20f;

	    if (t1 > 0.0f)
			t = t1;
	    if (t2 > 0.0f)
			t = fmin(t, t2);
	    if (t3 > 0.0f)
			t = fmin(t, t3);
	    if (t4 > 0.0f)
			t = fmin(t, t4);

		if (t < kTMin || t > ray.t_max_)
			return false;

		ray.t_max_ = t;
	    intersection.t = t;
		intersection.point = ray.position(intersection.t);
		intersection.normal = normalize(intersection.point * (dot(intersection.point, intersection.point) - radius_b_ * radius_b_ - radius_b_ * radius_b_ * make_float3(1.0f, 1.0f, -1.0f)));
		intersection.material = material_;
		return true;
	}

	__host__ __device__ void update(ObjectInfo* object_info, Material* material) override
	{
		const TorusInfo* torus_info = (TorusInfo*)object_info;
		center_ = torus_info->center.str;
		radius_a_ = torus_info->radius_a;
		radius_b_ = torus_info->radius_b;
		material_ = material;
	}

	__host__ __device__ Boundary bound() override
	{
		const float size = radius_a_ < radius_b_ ? radius_b_ : radius_a_;
		return {-make_float3(size), make_float3(size)};
	}

private:
	float3 center_{};
	float radius_a_{}, radius_b_{};
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

	__host__ __device__ bool intersect(const Ray& ray, Intersection& intersection, uint32_t* random_state) const override
	{
		Intersection temp_intersection{};
		bool intersected = false;

		for (uint64_t i = 0; i < triangle_count_; i++)
		{
			if (triangles_[i].intersect(ray, temp_intersection, random_state))
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