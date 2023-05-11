#pragma once
#include "../common/Camera.cuh"
#include "../common/Color.cuh"
#include "../info/RenderInfo.hpp"

__global__ void render_pixel_progressive(float4* frame_buffer, float4* accumulation_buffer, Camera** camera, World** world, SkyInfo sky_info, RenderInfo render_info, uint4* xoshiro_state);
__global__ void render_pixel_static(float4* frame_buffer, Camera** camera, World** world, SkyInfo sky_info, RenderInfo render_info, uint4* xoshiro_state);
__global__ void random_init(uint32_t max_x, uint32_t max_y, uint4* xoshiro_state);
__global__ void update_camera(Camera** camera, RenderInfo render_info);
__global__ void update_texture(World** world, int32_t index, TextureInfo** texture_data);
__global__ void update_material(World** world, int32_t index, MaterialInfo** material_data);
__global__ void update_object(World** world, int32_t index, ObjectInfo** object_data);
__global__ void create_world(ObjectInfo** object_data, MaterialInfo** material_data, TextureInfo** texture_data, int32_t object_count, int32_t material_count, int32_t texture_count, World** world);
__global__ void create_camera(Camera** camera, RenderInfo render_info);
__global__ void delete_world(World** world);
__global__ void delete_camera(Camera** camera);