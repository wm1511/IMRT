#include "CudaRenderer.cuh"
#include "Material.cuh"

#include <device_launch_parameters.h>
#include <cfloat>
#include <cstdio>

#define CCE(val) check_cuda( (val), #val, __FILE__, __LINE__ )

__host__ void check_cuda(const cudaError_t result, char const* const func, const char* const file, int const line)
{
    if (result)
    {
	    fprintf_s(stderr, "CUDA error = %u at %s: %i '%s' \n", (int32_t)result, file, line, func);
        cudaDeviceReset();
        abort();
    }
}

__device__ float3 calculate_color(const Ray& ray, World** world, const RenderInfo render_info, uint32_t* random_state)
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
            else return make_float3(0.0f);
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

__global__ void render_init(const uint32_t max_x, const uint32_t max_y, uint32_t* random_state)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= max_x || j >= max_y) return;
	const uint32_t pixel_index = j * max_x + i;
    random_state[pixel_index] = pixel_index;
}

__global__ void render_pixel(float4* frame_buffer, const uint32_t max_x, const uint32_t max_y, Camera** camera, World** world, const RenderInfo render_info, uint32_t* random_state)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= max_x || j >= max_y) return;
	const uint32_t pixel_index = j * max_x + i;
    uint32_t local_random_state = random_state[pixel_index];
    const int32_t samples_per_pixel = render_info.samples_per_pixel;

    float3 color = make_float3(0.0f);
    for (int32_t s = 0; s < samples_per_pixel; s++)
    {
	    const float u = ((float)i + pcg_rxs_m_xs(&local_random_state)) / (float)max_x;
	    const float v = ((float)j + pcg_rxs_m_xs(&local_random_state)) / (float)max_y;
        Ray ray = (*camera)->cast_ray(&local_random_state, u, v);
        color += calculate_color(ray, world, render_info, &local_random_state);
    }
    random_state[pixel_index] = local_random_state;
    color /= (float)samples_per_pixel;
    color = sqrt(color);
    frame_buffer[pixel_index] = make_float4(color, 1.0f);
}

__global__ void create_world(const RenderInfo render_info, MaterialInfo** material_data, ObjectInfo** object_data, Material** materials_list, Primitive** primitives_list, World** world)
{
	 if (threadIdx.x == 0 && blockIdx.x == 0) 
     {
         for (int32_t i = 0; i < render_info.material_count; i++)
         {
            if (material_data[i]->type == DIFFUSE)
            	materials_list[i] = new Diffuse((DiffuseInfo*)material_data[i]);
            else if (material_data[i]->type == SPECULAR)
            	materials_list[i] = new Specular((SpecularInfo*)material_data[i]);
            else if (material_data[i]->type == REFRACTIVE)
				materials_list[i] = new Refractive((RefractiveInfo*)material_data[i]);
         }

         for (int32_t i = 0; i < render_info.object_count; i++)
         {
            if (object_data[i]->type == SPHERE)
            	primitives_list[i] = new Sphere((SphereInfo*)object_data[i], materials_list[object_data[i]->material_id]);
            else if (object_data[i]->type == TRIANGLE)
            	primitives_list[i] = new Triangle((TriangleInfo*)object_data[i], materials_list[object_data[i]->material_id]);
         }

        *world = new World(primitives_list, render_info.object_count);
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

__global__ void delete_world(MaterialInfo** material_data, ObjectInfo** object_data, Material** materials, Primitive** primitives, World** world, const uint32_t material_count, const uint32_t primitive_count)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
	    for (uint32_t i = 0; i < material_count; i++)
	    {
	    	delete material_data[i];
	    	delete materials[i];
	    }

    	for (uint32_t i = 0; i < primitive_count; i++)
    	{
    		delete object_data[i];
    		delete primitives[i];
    	}
    
    	delete *world;
    }
}

__global__ void delete_camera(Camera** camera)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
	    delete *camera;
}

CudaRenderer::CudaRenderer(const RenderInfo* render_info, MaterialInfo** material_data, ObjectInfo** object_data, const uint32_t width, const uint32_t height)
		: render_info_(render_info)
{
    CCE(cudaMalloc((void**)&frame_buffer_, sizeof(float4) * width * height));
    CCE(cudaMalloc((void**)&random_state_, sizeof(uint32_t) * width * height));

    host_material_data_ = new MaterialInfo*[render_info_->material_count];
    for (int32_t i = 0; i < render_info_->material_count; i++)
    {
        if (material_data[i]->type == DIFFUSE)
        {
	        CCE(cudaMalloc((void**)&host_material_data_[i], sizeof(DiffuseInfo)));
			CCE(cudaMemcpy(host_material_data_[i], material_data[i], sizeof(DiffuseInfo), cudaMemcpyHostToDevice));
        }
        else if (material_data[i]->type == SPECULAR)
        {
	        CCE(cudaMalloc((void**)&host_material_data_[i], sizeof(SpecularInfo)));
			CCE(cudaMemcpy(host_material_data_[i], material_data[i], sizeof(SpecularInfo), cudaMemcpyHostToDevice));
        }
        else if (material_data[i]->type == REFRACTIVE)
        {
	        CCE(cudaMalloc((void**)&host_material_data_[i], sizeof(RefractiveInfo)));
			CCE(cudaMemcpy(host_material_data_[i], material_data[i], sizeof(RefractiveInfo), cudaMemcpyHostToDevice));
        }
		
    }
    host_object_data_ = new ObjectInfo*[render_info_->object_count];
    for (int32_t i = 0; i < render_info_->object_count; i++)
    {
        if (object_data[i]->type == SPHERE)
        {
	        CCE(cudaMalloc((void**)&host_object_data_[i], sizeof(SphereInfo)));
			CCE(cudaMemcpy(host_object_data_[i], object_data[i], sizeof(SphereInfo), cudaMemcpyHostToDevice));
        }
        else if (object_data[i]->type == TRIANGLE)
        {
	        CCE(cudaMalloc((void**)&host_object_data_[i], sizeof(TriangleInfo)));
			CCE(cudaMemcpy(host_object_data_[i], object_data[i], sizeof(TriangleInfo), cudaMemcpyHostToDevice));
        }
    }
    CCE(cudaMalloc((void**)&device_material_data_, render_info_->material_count * sizeof(MaterialInfo*)));
    CCE(cudaMalloc((void**)&device_object_data_, render_info_->object_count * sizeof(ObjectInfo*)));
	CCE(cudaMemcpy(device_material_data_, host_material_data_, render_info_->material_count * sizeof(MaterialInfo*), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(device_object_data_, host_object_data_, render_info_->object_count * sizeof(ObjectInfo*), cudaMemcpyHostToDevice));

    CCE(cudaMalloc((void**)&primitives_list_, render_info_->object_count * sizeof(Primitive*)));
    CCE(cudaMalloc((void**)&materials_list_, render_info_->material_count * sizeof(Material*)));
    CCE(cudaMalloc((void**)&world_, sizeof(World*)));
    CCE(cudaMalloc((void**)&camera_, sizeof(Camera*)));

    create_world<<<1, 1>>>(*render_info_, device_material_data_, device_object_data_, materials_list_, primitives_list_, world_);
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
    CCE(cudaFree(camera_));

    // TODO Fix error appearing after uncommenting line below
    CCE(cudaDeviceSynchronize());
    //delete_world<<<1, 1>>>(device_material_data_, device_object_data_, materials_list_, primitives_list_, world_, render_info_->material_count, render_info_->object_count);
    CCE(cudaGetLastError());
    CCE(cudaFree(world_));

    CCE(cudaFree(primitives_list_));
    CCE(cudaFree(materials_list_));

    for (int32_t i = 0; i < render_info_->object_count; i++)
	    CCE(cudaFree(host_object_data_[i]));
    for (int32_t i = 0; i < render_info_->material_count; i++)
		CCE(cudaFree(host_material_data_[i]));
    CCE(cudaFree(device_object_data_));
    CCE(cudaFree(device_material_data_));
    delete[] host_object_data_;
    delete[] host_material_data_;

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

    CCE(cudaMemcpy(image_data, frame_buffer_, sizeof(float4) * width * height, cudaMemcpyDeviceToHost));
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
    CCE(cudaMalloc((void**)&frame_buffer_, sizeof(float4) * width * height));
    CCE(cudaMalloc((void**)&random_state_, sizeof(uint32_t) * width * height));
}

void CudaRenderer::recreate_world(MaterialInfo** material_data, ObjectInfo** object_data)
{
    // TODO Fix error appearing after uncommenting line below
    CCE(cudaDeviceSynchronize());
    //delete_world<<<1, 1>>>(device_material_data_, device_object_data_, materials_list_, primitives_list_, world_, render_info_->material_count, render_info_->object_count);
    CCE(cudaGetLastError());
    CCE(cudaFree(world_));

    CCE(cudaFree(primitives_list_));
    CCE(cudaFree(materials_list_));

    for (int32_t i = 0; i < render_info_->object_count; i++)
	    CCE(cudaFree(host_object_data_[i]));
    for (int32_t i = 0; i < render_info_->material_count; i++)
		CCE(cudaFree(host_material_data_[i]));
    CCE(cudaFree(device_object_data_));
    CCE(cudaFree(device_material_data_));
    delete[] host_object_data_;
    delete[] host_material_data_;

    host_material_data_ = new MaterialInfo*[render_info_->material_count];
    for (int32_t i = 0; i < render_info_->material_count; i++)
    {
        if (material_data[i]->type == DIFFUSE)
        {
	        CCE(cudaMalloc((void**)&host_material_data_[i], sizeof(DiffuseInfo)));
			CCE(cudaMemcpy(host_material_data_[i], material_data[i], sizeof(DiffuseInfo), cudaMemcpyHostToDevice));
        }
        else if (material_data[i]->type == SPECULAR)
        {
	        CCE(cudaMalloc((void**)&host_material_data_[i], sizeof(SpecularInfo)));
			CCE(cudaMemcpy(host_material_data_[i], material_data[i], sizeof(SpecularInfo), cudaMemcpyHostToDevice));
        }
        else if (material_data[i]->type == REFRACTIVE)
        {
	        CCE(cudaMalloc((void**)&host_material_data_[i], sizeof(RefractiveInfo)));
			CCE(cudaMemcpy(host_material_data_[i], material_data[i], sizeof(RefractiveInfo), cudaMemcpyHostToDevice));
        }
		
    }
    host_object_data_ = new ObjectInfo*[render_info_->object_count];
    for (int32_t i = 0; i < render_info_->object_count; i++)
    {
        if (object_data[i]->type == SPHERE)
        {
	        CCE(cudaMalloc((void**)&host_object_data_[i], sizeof(SphereInfo)));
			CCE(cudaMemcpy(host_object_data_[i], object_data[i], sizeof(SphereInfo), cudaMemcpyHostToDevice));
        }
        else if (object_data[i]->type == TRIANGLE)
        {
	        CCE(cudaMalloc((void**)&host_object_data_[i], sizeof(TriangleInfo)));
			CCE(cudaMemcpy(host_object_data_[i], object_data[i], sizeof(TriangleInfo), cudaMemcpyHostToDevice));
        }
    }
    CCE(cudaMalloc((void**)&device_material_data_, render_info_->material_count * sizeof(MaterialInfo*)));
    CCE(cudaMalloc((void**)&device_object_data_, render_info_->object_count * sizeof(ObjectInfo*)));
	CCE(cudaMemcpy(device_material_data_, host_material_data_, render_info_->material_count * sizeof(MaterialInfo*), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(device_object_data_, host_object_data_, render_info_->object_count * sizeof(ObjectInfo*), cudaMemcpyHostToDevice));

    CCE(cudaMalloc((void**)&primitives_list_, render_info_->object_count * sizeof(Primitive*)));
    CCE(cudaMalloc((void**)&materials_list_, render_info_->material_count * sizeof(Material*)));
    CCE(cudaMalloc((void**)&world_, sizeof(World*)));
    CCE(cudaMalloc((void**)&camera_, sizeof(Camera*)));

    create_world<<<1, 1>>>(*render_info_, device_material_data_, device_object_data_, materials_list_, primitives_list_, world_);
    CCE(cudaGetLastError());
    CCE(cudaDeviceSynchronize());
}
