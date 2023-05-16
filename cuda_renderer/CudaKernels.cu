// ReSharper disable once CppPrecompiledHeaderIsNotIncluded
#include "CudaKernels.cuh"

#include <device_launch_parameters.h>

__global__ void render_pixel_progressive(float4* frame_buffer, float4* accumulation_buffer, World** world, SkyInfo sky_info, RenderInfo render_info, CameraInfo camera_info, uint4* xoshiro_state)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t j = threadIdx.y + blockIdx.y * blockDim.y;
	const uint32_t max_x = render_info.width;
	const uint32_t max_y = render_info.height;
    if (i >= max_x || j >= max_y) return;
	const uint32_t pixel_index = j * max_x + i;
    uint32_t local_random_state = xoshiro(&xoshiro_state[pixel_index]);

	const float u = ((float)i + pcg(&local_random_state)) / (float)max_x;
	const float v = ((float)j + pcg(&local_random_state)) / (float)max_y;
	const Ray ray = cast_ray(&local_random_state, u, v, camera_info);
    const float3 color = sqrt(calculate_color(ray, world, sky_info, render_info.max_depth, &local_random_state));

	accumulation_buffer[pixel_index] += make_float4(color, 1.0f);
	frame_buffer[pixel_index] = accumulation_buffer[pixel_index] / (float)render_info.frames_since_refresh;
}

__global__ void render_pixel_static(float4* frame_buffer, World** world, SkyInfo sky_info, RenderInfo render_info, CameraInfo camera_info, uint4* xoshiro_state)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t j = threadIdx.y + blockIdx.y * blockDim.y;
	const uint32_t max_x = render_info.width;
	const uint32_t max_y = render_info.height;
    if (i >= max_x || j >= max_y) return;
	const uint32_t pixel_index = j * max_x + i;
    uint32_t local_random_state = xoshiro(&xoshiro_state[pixel_index]);

	const float u = ((float)i + pcg(&local_random_state)) / (float)max_x;
	const float v = ((float)j + pcg(&local_random_state)) / (float)max_y;
	const Ray ray = cast_ray(&local_random_state, u, v, camera_info);
    const float3 color = sqrt(calculate_color(ray, world, sky_info, render_info.max_depth, &local_random_state));

	frame_buffer[pixel_index] += make_float4(color, 1.0f) / (float)render_info.samples_per_pixel;
}

__global__ void random_init(const uint32_t max_x, const uint32_t max_y, uint4* xoshiro_state)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= max_x || j >= max_y) return;
	const uint32_t pixel_index = j * max_x + i;
    xoshiro_state[pixel_index] = make_uint4(
        pixel_index + 15072003,
        pixel_index + 15112001,
        pixel_index + 10021151,
        pixel_index + 30027051);
}

__global__ void update_texture(World** world, const int32_t index, TextureInfo** texture_data)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
		(*world)->update_texture(index, texture_data[index]);
}

__global__ void update_material(World** world, const int32_t index, MaterialInfo** material_data)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
		(*world)->update_material(index, material_data[index]);
}

__global__ void update_object(World** world, const int32_t index, ObjectInfo** object_data)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) 
		(*world)->update_object(index, object_data[index]);
}

__global__ void create_world(ObjectInfo** object_data, MaterialInfo** material_data, TextureInfo** texture_data, const int32_t object_count, const int32_t material_count, const int32_t texture_count, World** world)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) 
       *world = new World(object_data, material_data, texture_data, object_count, material_count, texture_count);
}

__global__ void delete_world(World** world)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    	delete *world;
}