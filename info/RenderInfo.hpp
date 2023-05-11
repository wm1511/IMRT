#pragma once
#include <vector_types.h>

#include <cstdint>

enum class RenderMode
{
	PROGRESSIVE,
	STATIC
};

enum class RenderDevice
{
	CPU,
	CUDA,
	OPTIX
};

struct RenderInfo
{
	// Configuration
	RenderMode render_mode{RenderMode::PROGRESSIVE};
	RenderDevice render_device{RenderDevice::CPU};
	// Frame
	void* frame_handle = nullptr;
	uint64_t frame_size{};
	float* frame_data = nullptr;
	// Quality
	int32_t samples_per_pixel{100}, max_depth{10};
	uint32_t height = 0, width = 0, frames_since_refresh{0};
	// Camera
	float3 camera_position{0.0f, 0.0f, -4.0f}, camera_direction{0.0f, 0.0f, -1.0f};
	float fov{1.5f}, aperture{0.001f}, focus_distance{1.0f}, angle_x{0.0f}, angle_y{0.0f};
};