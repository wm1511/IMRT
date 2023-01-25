#pragma once
#include "../scene/ObjectInfo.hpp"
#include "../scene/MaterialInfo.hpp"

#include <cstdint>

struct RenderInfo
{
	// World
	int32_t object_data_count{0}, material_data_count{0}, object_count{0}, material_count{0}, object_capacity{0}, material_capacity{0};
	ObjectInfo** object_data = nullptr;
	MaterialInfo** material_data = nullptr;
	// Quality
	int32_t samples_per_pixel{4}, max_depth{8};
	uint32_t height = 0, width = 0, frames_since_refresh{0};
	// Camera
	float look_origin[3]{0.0f, 0.0f, 2.5f}, look_target[3]{0.375f, -0.5f, 0.0f}; 
	float fov{1.5f}, aperture{0.0f}, focus_distance{10.0f}, rr_stop_probability{0.1f};
};