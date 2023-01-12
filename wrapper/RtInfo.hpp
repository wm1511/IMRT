#pragma once
#include <cstdint>

struct RtInfo
{
	int32_t scene_index{0}, trace_type{0}, samples_per_pixel{8}, max_depth{10}, rr_certain_depth{5};
	float look_origin_x{0.0}, look_origin_y{0.0}, look_origin_z{0.0}, look_target_x{0.0}, look_target_y{-0.5}, look_target_z{-2.5};
	float fov{1.5f}, aperture{0.0f}, focus_distance{10.0f}, rr_stop_probability{0.1f};
};
