#pragma once
#include <vector_types.h>

#include <cstdint>

enum RenderMode
{
	PROGRESSIVE,
	STATIC
};

struct RenderInfo
{
	// Image
	int64_t image_handle{-1};
	uint64_t image_size{};
	// Quality
	int32_t samples_per_pixel{100}, max_depth{10}, render_mode{0};
	uint32_t height = 0, width = 0, frames_since_refresh{0};
	// Camera
	float3 camera_position{0.0f, 0.0f, -4.0f}, camera_direction{0.0f, 0.0f, -1.0f};
	float fov{1.5f}, aperture{0.001f}, focus_distance{1.0f}, angle_x{0.0f}, angle_y{0.0f};
};