#pragma once
#include "Boundary.cuh"
#include "Intersection.cuh"

class Transform
{
public:
	__host__ __device__ explicit Transform() {}
	__host__ __device__ explicit Transform(const Matrix4X4& m) : m_(m), m_inv_(invert(m)) {}
	__host__ __device__ explicit Transform(const Matrix4X4& m, const Matrix4X4& m_inv) : m_(m), m_inv_(m_inv) {}

	__host__ __device__ [[nodiscard]] const Matrix4X4& get_matrix() const { return m_; }
    __host__ __device__ [[nodiscard]] const Matrix4X4& get_inverse_matrix() const { return m_inv_; }

	__host__ __device__ [[nodiscard]] float3 transform(const float3& v) const
	{
		return {
			m_.m[0][0] * v.x + m_.m[0][1] * v.y + m_.m[0][2] * v.z,
			m_.m[1][0] * v.x + m_.m[1][1] * v.y + m_.m[1][2] * v.z,
			m_.m[2][0] * v.x + m_.m[2][1] * v.y + m_.m[2][2] * v.z
		};
	}

	__host__ __device__ [[nodiscard]] Boundary transform(const Boundary& b) const
	{
		Boundary boundary(transform(make_float3(b.min_.x, b.min_.y, b.min_.z)));    
	    boundary = unite(boundary, transform(make_float3(b.max_.x, b.min_.y, b.min_.z)));
	    boundary = unite(boundary, transform(make_float3(b.min_.x, b.max_.y, b.min_.z)));
	    boundary = unite(boundary, transform(make_float3(b.min_.x, b.min_.y, b.max_.z)));
	    boundary = unite(boundary, transform(make_float3(b.min_.x, b.max_.y, b.max_.z)));
	    boundary = unite(boundary, transform(make_float3(b.max_.x, b.max_.y, b.min_.z)));
	    boundary = unite(boundary, transform(make_float3(b.max_.x, b.min_.y, b.max_.z)));
	    boundary = unite(boundary, transform(make_float3(b.max_.x, b.max_.y, b.max_.z)));
	    return boundary;
	}

	__host__ __device__ [[nodiscard]] Intersection transform(const Intersection& i) const
	{
		Intersection intersection{};
		// TODO Check t equality
		intersection.t = i.t;
        intersection.point = transform(i.point);
        intersection.normal = normalize(transform(i.normal));
        intersection.uv = i.uv;
		intersection.material = i.material;

		return intersection; 
	}

	__host__ __device__ [[nodiscard]] Ray transform(const Ray& r) const
    {
	    const float3 origin = transform(r.origin_);
	    const float3 direction = transform(r.direction_);
		return {origin, direction, r.t_max_};
    }

	__host__ __device__ Transform operator*(const Transform& t) const
	{
		return Transform(multiply(m_, t.m_), multiply(t.m_inv_, m_inv_));
	}

private:
	Matrix4X4 m_{};
	Matrix4X4 m_inv_{};
};

inline __host__ __device__ Transform invert(const Transform& t)
{
    return Transform(t.get_inverse_matrix(), t.get_matrix());
}

inline __host__ __device__ Transform transpose(const Transform& t)
{
    return Transform(transpose(t.get_matrix()), transpose(t.get_inverse_matrix()));
}

inline __host__ __device__ Transform translate(const float3& t)
{
	const Matrix4X4 m = make_matrix4X4(
		1.0f, 0.0f, 0.0f, t.x,
		0.0f, 1.0f, 0.0f, t.y,
		0.0f, 0.0f, 1.0f, t.z,
		0.0f, 0.0f, 0.0f, 1.0f);

	const Matrix4X4 m_inv = make_matrix4X4(
		1.0f, 0.0f, 0.0f, -t.x, 
		0.0f, 1.0f, 0.0f, -t.y,
		0.0f, 0.0f, 1.0f, -t.z,
		0.0f, 0.0f, 0.0f, 1.0f);

    return Transform(m, m_inv);
}

inline __host__ __device__ Transform scale(const float3& s)
{
	const Matrix4X4 m = make_matrix4X4(
		s.x, 0.0f, 0.0f, 0.0f,
		0.0f, s.y, 0.0f, 0.0f,
		0.0f, 0.0f, s.z, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);

	const Matrix4X4 m_inv = make_matrix4X4(
		1.0f / s.x, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f / s.y, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f / s.z, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);

    return Transform(m, m_inv);
}

inline __host__ __device__ Transform rotate_x(const float x)
{
	const float sx = sin(x);
	const float cx = cos(x);

	const Matrix4X4 m = make_matrix4X4(
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, cx, -sx, 0.0f,
		0.0f, sx, cx, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);

    return Transform(m, transpose(m));
}

inline __host__ __device__ Transform rotate_y(const float y)
{
	const float sy = sin(y);
	const float cy = cos(y);

	const Matrix4X4 m = make_matrix4X4(
		cy, 0.0f, sy, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		-sy, 0.0f, cy, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);

    return Transform(m, transpose(m));
}

inline __host__ __device__ Transform rotate_z(const float z)
{
	const float sz = sin(z);
	const float cz = cos(z);

	const Matrix4X4 m = make_matrix4X4(
		cz, -sz, 0.0f, 0.0f,
		sz, cz, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);

    return Transform(m, transpose(m));
}

inline __host__ __device__ Transform look_at(const float3& position, const float3& direction, const float3& up)
{
    Matrix4X4 camera_to_world;

    camera_to_world.m[0][3] = position.x;
    camera_to_world.m[1][3] = position.y;
    camera_to_world.m[2][3] = position.z;
    camera_to_world.m[3][3] = 1;

    const float3 target = normalize(direction);
    const float3 u = normalize(cross(normalize(up), target));
    const float3 v = cross(target, u);

    camera_to_world.m[0][0] = u.x;
    camera_to_world.m[1][0] = u.y;
    camera_to_world.m[2][0] = u.z;
    camera_to_world.m[3][0] = 0.;
    camera_to_world.m[0][1] = v.x;
    camera_to_world.m[1][1] = v.y;
    camera_to_world.m[2][1] = v.z;
    camera_to_world.m[3][1] = 0.;
    camera_to_world.m[0][2] = target.x;
    camera_to_world.m[1][2] = target.y;
    camera_to_world.m[2][2] = target.z;
    camera_to_world.m[3][2] = 0.;

    return Transform(invert(camera_to_world), camera_to_world);
}

inline __host__ __device__ Transform perspective(const float fov, const float n, const float f)
{
    const Matrix4X4 perspective = make_matrix4X4(
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, f / (f - n), -f * n / (f - n),
		0.0f, 0.0f, 1.0f, 0.0f);

    const float viewport = 1.0f / tan(fov / 2.0f);
    return scale(make_float3(viewport, viewport, 1.0f)) * Transform(perspective);
}