#pragma once
#include "../common/Math.cuh"

#include <vector_functions.h>
#include <vector_types.h>

struct CameraInfo
{
	void update(const float width, const float height)
	{
		aspect_ratio = width / height;

		const float3 up = make_float3(0.0f, -1.0f, 0.0f);
		const float viewport_height = 2.0f * tan(fov * 0.5f);
		const float viewport_width = viewport_height * aspect_ratio;
		const float3 target = normalize(direction);

		u = versor(cross(up, target));
		v = cross(target, u);
		horizontal_map = focus_distance * viewport_width * u;
		vertical_map = focus_distance * viewport_height * v;
		starting_point = position - horizontal_map * 0.5f - vertical_map * 0.5f - focus_distance * target;
	}

	float3 position{0.0f, 0.0f, -4.0f}, direction{0.0f, 0.0f, -1.0f};
	float fov{1.5f}, focus_distance{1.0f}, angle_x{0.0f}, angle_y{0.0f}, lens_radius{0.001f};

	float3 starting_point{}, horizontal_map{}, vertical_map{}, u{}, v{};
	float aspect_ratio{};
};
