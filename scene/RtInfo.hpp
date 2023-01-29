#pragma once
#include "../scene/ObjectInfo.hpp"
#include "../scene/MaterialInfo.hpp"

#include <cstdint>

enum RenderMode
{
	PROGRESSIVE,
	STATIC
};

struct RenderInfo
{
	// World
	int32_t object_data_count{0}, material_data_count{0}, object_count{0}, material_count{0}, object_capacity{0}, material_capacity{0};
	ObjectInfo** object_data = nullptr;
	MaterialInfo** material_data = nullptr;
	// Environment
	float* hdr_data = nullptr;
	float hdr_exposure{2.0f};
	int32_t hdr_width{0}, hdr_height{0}, hdr_components{0};
	// Quality
	int32_t samples_per_pixel{4}, max_depth{8}, render_mode{0};
	uint32_t height = 0, width = 0, frames_since_refresh{0};
	// Camera
	float3 camera_position{1.0f, 0.0f, 2.0f}, camera_direction{0.0f, 0.0f, -1.0f}; 
	float fov{1.5f}, aperture{0.0f}, focus_distance{10.0f}, angle_x{3.2f}, angle_y{-0.17f};
};