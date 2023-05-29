#pragma once
#include <cstdint>

enum class RenderDevice
{
	CPU,
	CUDA,
	OPTIX
};

struct RenderInfo
{
	// Frame
	void* frame_handle = nullptr;
	uint64_t frame_size{};
	float* frame_data = nullptr;
	// Quality
	bool progressive{true};
	int32_t samples_per_pixel{100}, max_depth{10};
	uint32_t height = 0, width = 0, frames_since_refresh{0};
};