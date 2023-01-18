#include "CudaRenderer.cuh"
#include "Material.cuh"

#include <cfloat>
#include <cstdio>

#define CCE(val) check_cuda( (val), #val, __FILE__, __LINE__ )

__host__ void check_cuda(const cudaError_t result, char const* const func, const char* const file, int const line)
{
    if (result)
    {
	    fprintf(stderr, "CUDA error = %u at %s: %i '%s' \n", (uint32_t)result, file, line, func);
        cudaDeviceReset();
        abort();
    }
}

__device__ float3 calculate_color(const Ray& ray, World** world, const RenderInfo render_info, curandState* random_state)
{
	Ray current_ray = ray;
    float3 current_absorption = make_float3(1.0f);
    const int32_t max_depth = render_info.max_depth;

    for (int32_t i = 0; i < max_depth; i++)
    {
	    Intersection intersection{};
        if ((*world)->intersect(current_ray, 0.001f, FLT_MAX, intersection))
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

__global__ void render_init(const uint32_t max_x, const uint32_t max_y, curandState* random_state)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= max_x || j >= max_y) return;
	const uint32_t pixel_index = j * max_x + i;
    curand_init(1511 + pixel_index, 0, 0, &random_state[pixel_index]);
}

__global__ void render_pixel(float4* frame_buffer, const uint32_t max_x, const uint32_t max_y, Camera** camera, World** world, const RenderInfo render_info, curandState* random_state)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= max_x || j >= max_y) return;
	const uint32_t pixel_index = j * max_x + i;
    curandState local_random_state = random_state[pixel_index];
    const int32_t samples_per_pixel = render_info.samples_per_pixel;

    float3 color = make_float3(0.0f);
    for (int32_t s = 0; s < samples_per_pixel; s++)
    {
	    const float u = ((float)i + curand_uniform(&local_random_state)) / (float)max_x;
	    const float v = ((float)j + curand_uniform(&local_random_state)) / (float)max_y;
        Ray ray = (*camera)->cast_ray(&local_random_state, u, v);
        color += calculate_color(ray, world, render_info, &local_random_state);
    }
    random_state[pixel_index] = local_random_state;
    color /= (float)samples_per_pixel;
    color = sqrt(color);
    frame_buffer[pixel_index] = make_float4(color, 1.0f);
}

__global__ void create_world(Primitive** primitive_list, World** world)
{
	 if (threadIdx.x == 0 && blockIdx.x == 0) 
     {
        *primitive_list = new Sphere(make_float3(1.0f, 0.0f, -1.0f), 0.5f, new Diffuse(make_float3(0.5f)));
        *(primitive_list + 1) = new Sphere(make_float3(0.0f, 0.0f, -1.0f), -0.5f, new Refractive(0.5f));
        *(primitive_list + 2) = new Sphere(make_float3(-1.0f, 0.0f, -1.0f), 0.5f, new Specular(make_float3(0.5f), 0.1f));
        /**primitive_list = new Triangle(make_float3(-1.0f, -0.4f, -1.0f), make_float3(1.0f, -0.4f, -1.0f), make_float3(0.0f, 1.0f, -1.0f), new Refractive(0.5f));
        *(primitive_list + 1) = new Triangle(make_float3(1.0f, -0.4f, -1.1f), make_float3(-1.0f, -0.4f, -1.1f), make_float3(0.0f, 1.0f, -1.1f), new Refractive(0.5f));*/
	 	*(primitive_list + 3) = new Sphere(make_float3(0.0f, -100.5f,-1.0f), 100.0f, new Diffuse(make_float3(0.2f, 0.2f, 0.8f)));

	 	*world = new World(primitive_list, 4);
     }
}

__global__ void create_camera(Camera** camera, const uint32_t width, const uint32_t height, const RenderInfo render_info)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) 
    {
		*camera = new Camera(
	            make_float3(render_info.look_origin),
	            make_float3(render_info.look_target),
	            render_info.fov,
	            (float)width / (float)height,
	            render_info.aperture,
	            render_info.focus_distance);
    }
}

__global__ void delete_world(Primitive** primitive_list, World** world)
{
    for(uint32_t i = 0; i < 2; i++) 
    {
        delete primitive_list[i]->material_;
        delete primitive_list[i];
    }
    delete *world;
}

__global__ void delete_camera(Camera** camera)
{
    delete *camera;
}

CudaRenderer::CudaRenderer(const RenderInfo* render_info, const uint32_t width, const uint32_t height) : render_info_(render_info)
{
    CCE(cudaMalloc((void**)&frame_buffer_, width * height * sizeof(float4)));
    CCE(cudaMalloc((void**)&random_state_, width * height * sizeof(curandState)));

    constexpr int32_t primitives_count = 4;
    CCE(cudaMalloc((void**)&primitives_list_, primitives_count * sizeof(Primitive*)));
    CCE(cudaMalloc((void**)&world_, sizeof(Primitive*)));
    CCE(cudaMalloc((void**)&camera_, sizeof(Camera*)));

    create_world<<<1, 1>>>(primitives_list_, world_);
    CCE(cudaGetLastError());
    CCE(cudaDeviceSynchronize());

    create_camera<<<1, 1>>>(camera_, width, height, *render_info_);
    CCE(cudaGetLastError());
    CCE(cudaDeviceSynchronize());
}

CudaRenderer::~CudaRenderer()
{
    CCE(cudaDeviceSynchronize());
    delete_camera<<<1, 1>>>(camera_);
    CCE(cudaGetLastError());

    CCE(cudaDeviceSynchronize());
    // TODO Fix error appearing after uncommenting line below
    //delete_world<<<1, 1>>>(primitives_list_, world_);
    CCE(cudaGetLastError());

    CCE(cudaFree(camera_));
    CCE(cudaFree(world_));
    CCE(cudaFree(primitives_list_));
    CCE(cudaFree(random_state_));
    CCE(cudaFree(frame_buffer_));
    cudaDeviceReset();
}

void CudaRenderer::render(float* image_data, const uint32_t width, const uint32_t height)
{
    dim3 blocks((width + thread_x_ - 1) / thread_x_, (height + thread_y_ - 1) / thread_y_);
    dim3 threads(thread_x_, thread_y_);

    render_init<<<blocks, threads>>>(width, height, random_state_);
    CCE(cudaGetLastError());
    CCE(cudaDeviceSynchronize());
    render_pixel<<<blocks, threads>>>(frame_buffer_, width, height, camera_, world_, *render_info_, random_state_);
    CCE(cudaGetLastError());
    CCE(cudaDeviceSynchronize());

    CCE(cudaMemcpy(image_data, frame_buffer_, width * height * sizeof(float4), cudaMemcpyDeviceToHost));
}

void CudaRenderer::recreate_camera(const uint32_t width, const uint32_t height)
{
    CCE(cudaDeviceSynchronize());
    delete_camera<<<1, 1>>>(camera_);
    CCE(cudaGetLastError());

    create_camera<<<1, 1>>>(camera_, width, height, *render_info_);
    CCE(cudaGetLastError());
    CCE(cudaDeviceSynchronize());
}

void CudaRenderer::recreate_image(const uint32_t width, const uint32_t height)
{
    CCE(cudaFree(random_state_));
    CCE(cudaFree(frame_buffer_));
    CCE(cudaMalloc((void**)&frame_buffer_, width * height * sizeof(float4)));
    CCE(cudaMalloc((void**)&random_state_, width * height * sizeof(curandState)));
}

void CudaRenderer::recreate_scene()
{
}
