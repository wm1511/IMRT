#pragma once
#include "Ray.cuh"

class Camera
{
public:
	__host__ __device__ Camera(const float3 camera_position, const float3 camera_direction, const float fov, const float aspect_ratio, const float aperture, const float focus_distance) : aspect_ratio_(aspect_ratio)
	{
		update(camera_position, camera_direction, fov, aperture, focus_distance);
	}

	__host__ __device__ void update(const float3 camera_position, const float3 camera_direction, const float fov, const float aperture, const float focus_distance)
	{
		const float3 up = make_float3(0.0f, -1.0f, 0.0f);
		const float viewport_height = 2.0f * tan(fov * 0.5f);
		const float viewport_width = viewport_height * aspect_ratio_;
		const float3 target = normalize(camera_direction);

		origin_ = camera_position;
		lens_radius_ = aperture * 0.5f;

		u_ = versor(cross(up, target));
		v_ = cross(target, u_);
		horizontal_map_ = focus_distance * viewport_width * u_;
		vertical_map_ = focus_distance * viewport_height * v_;
		starting_point_ = origin_ - horizontal_map_ * 0.5f - vertical_map_ * 0.5f - focus_distance * target;
	}

	__host__ __device__ Ray cast_ray(uint32_t* random_state, const float screen_x, const float screen_y) const
	{
		const float2 random_on_lens = lens_radius_ * disk_random(random_state);
		const float3 ray_offset = u_ * random_on_lens.x + v_ * random_on_lens.y;
		return {origin_ + ray_offset, starting_point_ + screen_x * horizontal_map_ + screen_y * vertical_map_ - origin_ - ray_offset};
	}

private:
	float3 origin_{};
	float3 starting_point_{};
	float3 horizontal_map_{}, vertical_map_{};
	float3 u_{}, v_{};
	float lens_radius_{};
	float aspect_ratio_;
};