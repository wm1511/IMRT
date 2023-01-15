#pragma once
#include <cstdint>

struct RtInfo
{
	int32_t samples_per_pixel{8}, max_depth{10};
	float look_origin[3]{0.0f, 0.0f, 0.0f}, look_target[3]{0.0f, -0.5f, -2.5f}; 
	float fov{1.5f}, aperture{0.0f}, focus_distance{10.0f}, rr_stop_probability{0.1f};
};
