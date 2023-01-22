#pragma once
#include <cstdint>

struct RenderInfo
{
	// World
	int32_t object_count{0}, material_count{0}, object_capacity{0}, material_capacity{0};
	// Quality
	int32_t samples_per_pixel{8}, max_depth{10};
	// Camera
	float look_origin[3]{0.0f, 0.0f, 2.5f}, look_target[3]{0.375f, -0.5f, 0.0f}; 
	float fov{1.5f}, aperture{0.0f}, focus_distance{10.0f}, rr_stop_probability{0.1f};
};