#include "CudaRenderer.cuh"
#include "Ray.cuh"
#include "World.cuh"
#include "Material.cuh"
#include "Camera.cuh"

#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include <cfloat>
#include <cstdio>

#define CCE(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(const cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) 
    {
	    fprintf(stderr, "CUDA error = %u at %s: %i '%s' \n", (uint32_t)result, file, line, func);
        cudaDeviceReset();
        abort();
    }
}

__device__ float3 calculate_color(const Ray& ray, Primitive** primitives, const RtInfo rt_info, curandState* random_state)
{
	Ray current_ray = ray;
    float3 current_absorption = make_float3(1.0f);
    const int32_t max_depth = rt_info.max_depth;

    for (int32_t i = 0; i < max_depth; i++)
    {
	    Intersection intersection{};
        if ((*primitives)->intersect(current_ray, 0.001f, FLT_MAX, intersection))
        {
            Ray scattered_ray({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f});
            float3 absorption;
            if (intersection.material->scatter(current_ray, intersection, absorption, scattered_ray, random_state))
            {
	            current_absorption *= absorption;
                current_ray = scattered_ray;
            }
            else
            {
	            return make_float3(0.0f);
            }
        }
        else
        {
	        const float t = 0.5f * (versor(current_ray.direction()).y + 1.0f);
	        const float3 color = (1.0f - t) * make_float3(1.0f) + t * make_float3(0.5f, 0.7f, 1.0f);
            return current_absorption * color;
        }
    }
	return make_float3(0.0f);
}

__global__ void random_init(curandState* random_state)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		curand_init(1511, 0, 0, random_state);
	}
}

__global__ void render_init(const uint32_t max_x, const uint32_t max_y, curandState* random_state)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= max_x || j >= max_y) return;
	const uint32_t pixel_index = j * max_x + i;
    curand_init(1511 + pixel_index, 0, 0, &random_state[pixel_index]);
}

__global__ void render_pixel(float4* frame_buffer, const uint32_t max_x, const uint32_t max_y, Camera** camera, Primitive** primitives, const RtInfo rt_info, curandState* random_state)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= max_x || j >= max_y) return;
	const uint32_t pixel_index = j * max_x + i;
    curandState local_random_state = random_state[pixel_index];
    const int32_t samples_per_pixel = rt_info.samples_per_pixel;

    float3 color = make_float3(0.0f);
    for (int32_t s = 0; s < samples_per_pixel; s++)
    {
	    const float u = ((float)i + curand_uniform(&local_random_state)) / (float)max_x;
	    const float v = ((float)j + curand_uniform(&local_random_state)) / (float)max_y;
        Ray ray = (*camera)->cast_ray(&local_random_state, u, v);
        color += calculate_color(ray, primitives, rt_info, &local_random_state);
    }
    random_state[pixel_index] = local_random_state;
    color /= (float)samples_per_pixel;
    color = sqrt(color);
    frame_buffer[pixel_index] = make_float4(color, 1.0f);
}

__global__ void create_world(Primitive** primitive_list, Primitive** world, Camera** camera, const uint32_t width, const uint32_t height, const RtInfo rt_info, curandState* random_state)
{
	 if (threadIdx.x == 0 && blockIdx.x == 0) 
     {
        //curandState local_random_state = *random_state;
        *primitive_list = new Sphere(make_float3(0.0f, 0.0f, -1.0f), 0.5, new Diffuse(make_float3(0.5f)));
        *(primitive_list + 1) = new Sphere(make_float3(0.0f, -100.5f,-1.0f), 100.0f, new Diffuse(make_float3(0.2f, 0.2f, 0.8f)));

	 	*world = new World(primitive_list, 2);
        *camera = new Camera(
            make_float3(rt_info.look_origin),
            make_float3(rt_info.look_target),
            rt_info.fov,
            (float)width / (float)height,
            rt_info.aperture,
            rt_info.focus_distance);
        //*random_state = local_random_state;
     }
}

__global__ void delete_world(Primitive** primitive_list, Primitive** world, Camera** camera)
{
    for(uint32_t i = 0; i < 2; i++) 
    {
        delete ((Sphere*)primitive_list[i])->material_;
        delete primitive_list[i];
    }
    delete *world;
    delete *camera;
}

CudaRenderer::CudaRenderer(const RtInfo* rt_info) : rt_info_(rt_info)
{
}

void CudaRenderer::render(float* image_data, const uint32_t width, const uint32_t height)
{
	constexpr int32_t thread_x = 16;
	constexpr int32_t thread_y = 16;
    const uint32_t pixel_count = width * height;
    const uint32_t frame_buffer_size = pixel_count * sizeof(float4);

    float4* frame_buffer;
    CCE(cudaMallocManaged((void**)&frame_buffer, frame_buffer_size));

    curandState* random_state;
    CCE(cudaMalloc((void**)&random_state, pixel_count * sizeof(curandState)));
    curandState* random_state2;
    CCE(cudaMalloc((void**)&random_state2, sizeof(curandState)));

    random_init<<<1, 1>>>(random_state2);
    CCE(cudaGetLastError());
    CCE(cudaDeviceSynchronize());

    Primitive** primitives_list;
    constexpr int32_t primitives_count = 2;
    CCE(cudaMalloc((void**)&primitives_list, primitives_count * sizeof(Primitive*)));
    Primitive** world;
    CCE(cudaMalloc((void**)&world, sizeof(Primitive*)));
    Camera** camera;
    CCE(cudaMalloc((void**)&camera, sizeof(Camera*)));
    create_world<<<1, 1>>>(primitives_list, world, camera, width, height, *rt_info_, random_state2);
    CCE(cudaGetLastError());
    CCE(cudaDeviceSynchronize());

    dim3 blocks(width / thread_x + 1, height / thread_y + 1);
    dim3 threads(thread_x, thread_y);
    render_init<<<blocks, threads>>>(width, height, random_state);
    CCE(cudaGetLastError());
    CCE(cudaDeviceSynchronize());
    render_pixel<<<blocks, threads>>>(frame_buffer, width, height, camera, world, *rt_info_, random_state);
    CCE(cudaGetLastError());
    CCE(cudaDeviceSynchronize());

    CCE(cudaMemcpy(image_data, frame_buffer, frame_buffer_size, cudaMemcpyDeviceToHost));

    CCE(cudaDeviceSynchronize());
    // TODO Fix error appearing after uncommenting line below
    //delete_world<<<1, 1>>>(primitives_list, world, camera);
    CCE(cudaGetLastError());

    CCE(cudaFree(camera));
    CCE(cudaFree(world));
    CCE(cudaFree(primitives_list));
    CCE(cudaFree(random_state));
    CCE(cudaFree(random_state2));
    CCE(cudaFree(frame_buffer));
    cudaDeviceReset();
}