#pragma once
#include "../common/Color.hpp"
#include "../info/RenderInfo.hpp"

__global__ void render_pixel_progressive(float4* frame_buffer, float4* accumulation_buffer, const World* world, SkyInfo sky_info, RenderInfo render_info, CameraInfo camera_info, uint4* xoshiro_state);
__global__ void render_pixel_static(float4* frame_buffer, const World* world, SkyInfo sky_info, RenderInfo render_info, CameraInfo camera_info, uint4* xoshiro_state);
__global__ void random_init(uint32_t max_x, uint32_t max_y, uint4* xoshiro_state);