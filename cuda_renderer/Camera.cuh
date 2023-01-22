#pragma once
#include "Ray.cuh"

class Camera
{
public:
	__device__ Camera(const float3 look_origin, const float3 look_target, const float fov, const float aspect_ratio, const float aperture, const float focus_distance)
		: origin_(look_origin), lens_radius_(aperture / 2.0f)
	{
		const float viewport_height = 2.0f * tan(fov / 2.0f);
		const float viewport_width = viewport_height * aspect_ratio;
		const float3 camera_direction = normalize(look_origin - look_target);

		u_ = normalize(cross(make_float3(0.0f, -1.0f, 0.0f), camera_direction));
		v_ = cross(camera_direction, u_);
		horizontal_map_ = focus_distance * viewport_width * u_;
		vertical_map_ = focus_distance * viewport_height * v_;
		starting_point_ = origin_ - horizontal_map_ / 2.0f - vertical_map_ - 2.0f - focus_distance * camera_direction;
	}

	__device__ Ray cast_ray(uint32_t* random_state, const float screen_x, const float screen_y) const
	{
		const float3 random_on_lens = lens_radius_ * disk_random(random_state);
		const float3 ray_offset = u_ * random_on_lens.x + v_ * random_on_lens.y;
		return {origin_ + ray_offset, starting_point_ + screen_x * horizontal_map_ + screen_y * vertical_map_ - origin_ - ray_offset};
	}

private:
	float3 origin_;
	float3 starting_point_;
	float3 horizontal_map_, vertical_map_;
	float3 u_, v_;
	float lens_radius_;
};