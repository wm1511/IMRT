#include "CudaRenderer.cuh"

#include <device_launch_parameters.h>

#include <cstdio>

#define CCE(val) check_cuda( (val), #val, __FILE__, __LINE__ )

__host__ void check_cuda(const cudaError_t result, char const* const func, const char* const file, int const line)
{
    if (result)
    {
	    printf("CUDA error = %u at %s: %i '%s' \n", (int32_t)result, file, line, func);
        cudaDeviceReset();
        abort();
    }
}

__global__ void render_pixel(float4* frame_buffer, float4* accumulation_buffer, Camera** camera, World** world, const float3* hdr_data, const RenderInfo render_info, uint4* xoshiro_state)
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
	const Ray ray = (*camera)->cast_ray(&local_random_state, u, v);
    const float3 color = sqrt(calculate_color(ray, world, hdr_data, render_info, &local_random_state));

    if (render_info.render_mode == PROGRESSIVE)
    {
    	accumulation_buffer[pixel_index] += make_float4(color, 1.0f);
    	frame_buffer[pixel_index] = accumulation_buffer[pixel_index] / (float)render_info.frames_since_refresh;
    }
    else if (render_info.render_mode == STATIC)
    {
        frame_buffer[pixel_index] += make_float4(color, 1.0f) / (float)render_info.samples_per_pixel;
    }
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

__global__ void update_world(const int32_t material_count, const int32_t object_count, MaterialInfo** material_data, Material** materials_list, ObjectInfo** object_data, Primitive** primitives_list, World** world)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) 
    {
        for (int32_t i = 0; i < material_count; i++)
        {
           if (material_data[i]->type == DIFFUSE)
               ((Diffuse*)materials_list[i])->update((DiffuseInfo*)material_data[i]);
           else if (material_data[i]->type == SPECULAR)
               ((Specular*)materials_list[i])->update((SpecularInfo*)material_data[i]);
           else if (material_data[i]->type == REFRACTIVE)
    		((Specular*)materials_list[i])->update((SpecularInfo*)material_data[i]);
        }
    
		for (int32_t i = 0; i < object_count; i++)
        {
           if (object_data[i]->type == SPHERE)
               ((Sphere*)primitives_list[i])->update((SphereInfo*)object_data[i], materials_list[object_data[i]->material_id]);
           else if (object_data[i]->type == TRIANGLE)
               ((Triangle*)primitives_list[i])->update((TriangleInfo*)object_data[i], materials_list[object_data[i]->material_id]);
        }

        (*world)->update(object_count);
    }
}

__global__ void update_camera(Camera** camera, const RenderInfo render_info)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) 
    {
		(*camera)->update(
			render_info.camera_position,
	        render_info.camera_direction,
	        render_info.fov,
			render_info.aperture,
	        render_info.focus_distance);
    }
}

__global__ void create_world(const int32_t material_count, const int32_t object_count, MaterialInfo** material_data, ObjectInfo** object_data, Material** materials_list, Primitive** primitives_list, World** world)
{
	 if (threadIdx.x == 0 && blockIdx.x == 0) 
     {
         for (int32_t i = 0; i < material_count; i++)
         {
            if (material_data[i]->type == DIFFUSE)
            	materials_list[i] = new Diffuse((DiffuseInfo*)material_data[i]);
            else if (material_data[i]->type == SPECULAR)
            	materials_list[i] = new Specular((SpecularInfo*)material_data[i]);
            else if (material_data[i]->type == REFRACTIVE)
				materials_list[i] = new Refractive((RefractiveInfo*)material_data[i]);
         }

         for (int32_t i = 0; i < object_count; i++)
         {
            if (object_data[i]->type == SPHERE)
            	primitives_list[i] = new Sphere((SphereInfo*)object_data[i], materials_list[object_data[i]->material_id]);
            else if (object_data[i]->type == TRIANGLE)
            	primitives_list[i] = new Triangle((TriangleInfo*)object_data[i], materials_list[object_data[i]->material_id]);
         }

        *world = new World(primitives_list, object_count);
     }
}

__global__ void create_camera(Camera** camera, const RenderInfo render_info)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) 
    {
		*camera = new Camera(
	            render_info.camera_position,
	            render_info.camera_direction,
	            render_info.fov,
	            (float)render_info.width / (float)render_info.height,
	            render_info.aperture,
	            render_info.focus_distance);
    }
}

__global__ void delete_world(Material** materials_list, Primitive** primitives_list, World** world, const uint32_t material_count, const uint32_t primitive_count)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
	    for (uint32_t i = 0; i < material_count; i++)
	    	delete materials_list[i];

    	for (uint32_t i = 0; i < primitive_count; i++)
    		delete primitives_list[i];
    	
    	delete *world;
    }
}

__global__ void delete_camera(Camera** camera)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
	    delete *camera;
}

CudaRenderer::CudaRenderer(const RenderInfo* render_info, const WorldInfo* world_info) : render_info_(render_info), world_info_(world_info)
{
    const uint32_t width = render_info_->width;
    const uint32_t height = render_info_->height;
    constexpr int32_t thread_x = 16;
	constexpr int32_t thread_y = 16;
    blocks_ = dim3((width + thread_x - 1) / thread_x, (height + thread_y - 1) / thread_y);
    threads_ = dim3(thread_x, thread_y);

    CCE(cudaMalloc((void**)&frame_buffer_, sizeof(float4) * width * height));
    CCE(cudaMalloc((void**)&accumulation_buffer_, sizeof(float4) * width * height));
    CCE(cudaMalloc((void**)&xoshiro_state_, sizeof(uint4) * width * height));

    random_init<<<blocks_, threads_>>>(width, height, xoshiro_state_);
    CCE(cudaGetLastError());
    CCE(cudaDeviceSynchronize());
    
    allocate_world();
    
    CCE(cudaMalloc((void**)&camera_, sizeof(Camera*)));
    create_camera<<<1, 1>>>(camera_, *render_info_);
    CCE(cudaGetLastError());
    CCE(cudaDeviceSynchronize());

    if (render_info_->hdr_data)
    {
        const uint64_t hdr_size = sizeof(float3) * render_info_->hdr_width * render_info_->hdr_height;
	    CCE(cudaMalloc((void**)&device_hdr_data_, hdr_size));
    	CCE(cudaMemcpy(device_hdr_data_, render_info_->hdr_data, hdr_size, cudaMemcpyHostToDevice));
    }
}

CudaRenderer::~CudaRenderer()
{
    if (render_info_->hdr_data)
	    CCE(cudaFree(device_hdr_data_));

    CCE(cudaDeviceSynchronize());
    delete_camera<<<1, 1>>>(camera_);
    CCE(cudaGetLastError());
    CCE(cudaFree(camera_));

    deallocate_world();

    CCE(cudaFree(xoshiro_state_));
    CCE(cudaFree(accumulation_buffer_));
    CCE(cudaFree(frame_buffer_));
    cudaDeviceReset();
}

void CudaRenderer::render(float* image_data)
{
    const uint32_t width = render_info_->width;
    const uint32_t height = render_info_->height;

    if (render_info_->render_mode == PROGRESSIVE)
	    render_pixel<<<blocks_, threads_>>>(frame_buffer_, accumulation_buffer_, camera_, world_, device_hdr_data_, *render_info_, xoshiro_state_);
    else if (render_info_->render_mode == STATIC)
	    for (int32_t i = 0; i < render_info_->samples_per_pixel; i++)
			render_pixel<<<blocks_, threads_>>>(frame_buffer_, accumulation_buffer_, camera_, world_, device_hdr_data_, *render_info_, xoshiro_state_);

	CCE(cudaGetLastError());
    CCE(cudaDeviceSynchronize());

    if (render_info_->frame_needs_display)
		CCE(cudaMemcpy(image_data, frame_buffer_, sizeof(float4) * width * height, cudaMemcpyDeviceToHost));
}

void CudaRenderer::refresh_buffer()
{
    const uint32_t width = render_info_->width;
    const uint32_t height = render_info_->height;

    CCE(cudaMemset(accumulation_buffer_, 0, sizeof(float4) * width * height));
    random_init<<<blocks_, threads_>>>(width, height, xoshiro_state_);
    CCE(cudaGetLastError());
    CCE(cudaDeviceSynchronize());
}

void CudaRenderer::refresh_camera()
{
    update_camera<<<1, 1>>>(camera_, *render_info_);
    CCE(cudaGetLastError());
    CCE(cudaDeviceSynchronize());
}

void CudaRenderer::refresh_world()
{
    reload_world();

    update_world<<<1, 1>>>(world_info_->material_count, world_info_->object_count, device_material_data_, materials_list_, device_object_data_, primitives_list_, world_);
    CCE(cudaGetLastError());
    CCE(cudaDeviceSynchronize());
}

void CudaRenderer::recreate_camera()
{
    CCE(cudaDeviceSynchronize());
    delete_camera<<<1, 1>>>(camera_);
    CCE(cudaGetLastError());

    create_camera<<<1, 1>>>(camera_, *render_info_);
    CCE(cudaGetLastError());
    CCE(cudaDeviceSynchronize());
}

void CudaRenderer::recreate_image()
{
    const uint32_t width = render_info_->width;
    const uint32_t height = render_info_->height;
	constexpr int32_t thread_x = 16;
	constexpr int32_t thread_y = 16;
    blocks_ = dim3((width + thread_x - 1) / thread_x, (height + thread_y - 1) / thread_y);
    threads_ = dim3(thread_x, thread_y);

    CCE(cudaFree(frame_buffer_));
    CCE(cudaFree(xoshiro_state_));
    CCE(cudaFree(accumulation_buffer_));
    CCE(cudaMalloc((void**)&accumulation_buffer_, sizeof(float4) * width * height));
    CCE(cudaMalloc((void**)&xoshiro_state_, sizeof(uint4) * width * height));
    CCE(cudaMalloc((void**)&frame_buffer_, sizeof(float4) * width * height));
}

void CudaRenderer::recreate_sky()
{
    CCE(cudaFree(device_hdr_data_));

    if (render_info_->hdr_data)
    {
	    const uint64_t hdr_size = sizeof(float3) * render_info_->hdr_width * render_info_->hdr_height;
    	CCE(cudaMalloc((void**)&device_hdr_data_, hdr_size));
    	CCE(cudaMemcpy(device_hdr_data_, render_info_->hdr_data, hdr_size, cudaMemcpyHostToDevice));
    }
    else
        device_hdr_data_ = nullptr;
}

void CudaRenderer::allocate_world()
{
    MaterialInfo** material_data = world_info_->material_data;
    ObjectInfo** object_data = world_info_->object_data;
    host_material_data_ = new MaterialInfo*[world_info_->material_count];
    host_object_data_ = new ObjectInfo*[world_info_->object_count];

    for (int32_t i = 0; i < world_info_->material_count; i++)
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
    for (int32_t i = 0; i < world_info_->object_count; i++)
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

    CCE(cudaMalloc((void**)&device_material_data_, world_info_->material_count * sizeof(MaterialInfo*)));
    CCE(cudaMalloc((void**)&device_object_data_, world_info_->object_count * sizeof(ObjectInfo*)));
	CCE(cudaMemcpy(device_material_data_, host_material_data_, world_info_->material_count * sizeof(MaterialInfo*), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(device_object_data_, host_object_data_, world_info_->object_count * sizeof(ObjectInfo*), cudaMemcpyHostToDevice));

    CCE(cudaMalloc((void**)&primitives_list_, world_info_->object_count * sizeof(Primitive*)));
    CCE(cudaMalloc((void**)&materials_list_, world_info_->material_count * sizeof(Material*)));
    CCE(cudaMalloc((void**)&world_, sizeof(World*)));

    create_world<<<1, 1>>>(world_info_->material_count, world_info_->object_count, device_material_data_, device_object_data_, materials_list_, primitives_list_, world_);
    CCE(cudaGetLastError());
    CCE(cudaDeviceSynchronize());
}

void CudaRenderer::deallocate_world() const
{
    CCE(cudaDeviceSynchronize());
	//delete_world<<<1, 1>>>(materials_list_, primitives_list_, world_, world_info_->material_count, world_info_->object_count);
    CCE(cudaGetLastError());
    CCE(cudaFree(world_));

    CCE(cudaFree(primitives_list_));
    CCE(cudaFree(materials_list_));

    for (int32_t i = 0; i < world_info_->object_count; i++)
	    CCE(cudaFree(host_object_data_[i]));
    for (int32_t i = 0; i < world_info_->material_count; i++)
		CCE(cudaFree(host_material_data_[i]));
    CCE(cudaFree(device_object_data_));
    CCE(cudaFree(device_material_data_));

    delete[] host_object_data_;
    delete[] host_material_data_;
}

void CudaRenderer::reload_world() const
{
    MaterialInfo** material_data = world_info_->material_data;
    ObjectInfo** object_data = world_info_->object_data;

    for (int32_t i = 0; i < world_info_->material_count; i++)
    {
        if (material_data[i]->type == DIFFUSE)
			CCE(cudaMemcpy(host_material_data_[i], material_data[i], sizeof(DiffuseInfo), cudaMemcpyHostToDevice));
        else if (material_data[i]->type == SPECULAR)
			CCE(cudaMemcpy(host_material_data_[i], material_data[i], sizeof(SpecularInfo), cudaMemcpyHostToDevice));
        else if (material_data[i]->type == REFRACTIVE)
			CCE(cudaMemcpy(host_material_data_[i], material_data[i], sizeof(RefractiveInfo), cudaMemcpyHostToDevice));
    }

    for (int32_t i = 0; i < world_info_->object_count; i++)
    {
        if (object_data[i]->type == SPHERE)
			CCE(cudaMemcpy(host_object_data_[i], object_data[i], sizeof(SphereInfo), cudaMemcpyHostToDevice));
        else if (object_data[i]->type == TRIANGLE)
			CCE(cudaMemcpy(host_object_data_[i], object_data[i], sizeof(TriangleInfo), cudaMemcpyHostToDevice));
    }

	CCE(cudaMemcpy(device_material_data_, host_material_data_, world_info_->material_count * sizeof(MaterialInfo*), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(device_object_data_, host_object_data_, world_info_->object_count * sizeof(ObjectInfo*), cudaMemcpyHostToDevice));
}
