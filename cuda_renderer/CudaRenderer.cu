//#include "stdafx.h"
#include "CudaRenderer.cuh"

#include <device_launch_parameters.h>

#include <cstdio>

#define CCE(val) check_cuda( (val), #val, __FILE__, __LINE__ )

__host__ void check_cuda(const cudaError_t result, char const* const func, const char* const file, int const line)
{
    if (result)
    {
	    printf("CUDA error = %u at %s: %i '%s' \n", static_cast<int32_t>(result), file, line, func);
        cudaDeviceReset();
        abort();
    }
}

__global__ void render_pixel_progressive(float4* frame_buffer, float4* accumulation_buffer, Camera** camera, World** world, const SkyInfo sky_info, const RenderInfo render_info, uint4* xoshiro_state)
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
    const float3 color = sqrt(calculate_color(ray, world, sky_info, render_info.max_depth, &local_random_state));

	accumulation_buffer[pixel_index] += make_float4(color, 1.0f);
	frame_buffer[pixel_index] = accumulation_buffer[pixel_index] / (float)render_info.frames_since_refresh;
}

__global__ void render_pixel_static(float4* frame_buffer, Camera** camera, World** world, const SkyInfo sky_info, const RenderInfo render_info, uint4* xoshiro_state)
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

__global__ void update_camera(Camera** camera, const RenderInfo render_info)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) 
		(*camera)->update(
			render_info.camera_position,
	        render_info.camera_direction,
			render_info.fov,
	        render_info.aperture,
	        render_info.focus_distance);
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

__global__ void create_camera(Camera** camera, const RenderInfo render_info)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) 
		*camera = new Camera(
	            render_info.camera_position,
	            render_info.camera_direction,
	            render_info.fov,
	            (float)render_info.width / (float)render_info.height,
	            render_info.aperture,
	            render_info.focus_distance);
}

__global__ void delete_world(World** world)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    	delete *world;
}

__global__ void delete_camera(Camera** camera)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
	    delete *camera;
}

CudaRenderer::CudaRenderer(const RenderInfo* render_info, const WorldInfo* world_info, SkyInfo* sky_info)
	: render_info_(render_info), world_info_(world_info), sky_info_(sky_info)
{
    const uint32_t width = render_info_->width;
    const uint32_t height = render_info_->height;
    constexpr int32_t thread_x = 16;
	constexpr int32_t thread_y = 16;
    blocks_ = dim3((width + thread_x - 1) / thread_x, (height + thread_y - 1) / thread_y);
    threads_ = dim3(thread_x, thread_y);

    CCE(cudaMalloc(reinterpret_cast<void**>(&accumulation_buffer_), sizeof(float4) * width * height));
    CCE(cudaMalloc(reinterpret_cast<void**>(&xoshiro_initial_), sizeof(uint4) * width * height));
    CCE(cudaMalloc(reinterpret_cast<void**>(&xoshiro_state_), sizeof(uint4) * width * height));

    random_init<<<blocks_, threads_>>>(width, height, xoshiro_initial_);
    CCE(cudaGetLastError());
    CCE(cudaDeviceSynchronize());

    CCE(cudaMemcpy(xoshiro_state_, xoshiro_initial_, sizeof(uint4) * width * height, cudaMemcpyDeviceToDevice));
    
    allocate_world();
    
    CCE(cudaMalloc(reinterpret_cast<void**>(&camera_), sizeof(Camera*)));
    create_camera<<<1, 1>>>(camera_, *render_info_);
    CCE(cudaGetLastError());
    CCE(cudaDeviceSynchronize());

    if (sky_info_->buffered_hdr_data)
    {
        const uint64_t hdr_size = sizeof(float3) * sky_info_->hdr_width * sky_info_->hdr_height;
	    CCE(cudaMalloc(reinterpret_cast<void**>(&sky_info_->usable_hdr_data), hdr_size));
    	CCE(cudaMemcpy(sky_info_->usable_hdr_data, sky_info_->buffered_hdr_data, hdr_size, cudaMemcpyHostToDevice));
    }
}

CudaRenderer::~CudaRenderer()
{
    if (sky_info_->buffered_hdr_data)
	    CCE(cudaFree(sky_info_->usable_hdr_data));

    CCE(cudaDeviceSynchronize());
    delete_camera<<<1, 1>>>(camera_);
    CCE(cudaGetLastError());
    CCE(cudaFree(camera_));

    deallocate_world();

    CCE(cudaFree(xoshiro_state_));
    CCE(cudaFree(xoshiro_initial_));
    CCE(cudaFree(accumulation_buffer_));
    cudaDeviceReset();
}

void CudaRenderer::render()
{
    fetch_frame_buffer();

    if (render_info_->render_mode == PROGRESSIVE)
    {
    	render_pixel_progressive<<<blocks_, threads_>>>(frame_buffer_, accumulation_buffer_, camera_, world_, *sky_info_, *render_info_, xoshiro_state_);
    }
    else if (render_info_->render_mode == STATIC)
    {
        CCE(cudaMemset(frame_buffer_, 0, render_info_->frame_size));

	    for (int32_t i = 0; i < render_info_->samples_per_pixel; i++)
	    	render_pixel_static<<<blocks_, threads_>>>(frame_buffer_, camera_, world_, *sky_info_, *render_info_, xoshiro_state_);
    }

	CCE(cudaGetLastError());
    CCE(cudaDeviceSynchronize());

    CCE(cudaFree(frame_buffer_));
}

void CudaRenderer::refresh_buffer()
{
    const uint32_t width = render_info_->width;
    const uint32_t height = render_info_->height;

    CCE(cudaMemset(accumulation_buffer_, 0, sizeof(float4) * width * height));
	CCE(cudaMemcpy(xoshiro_state_, xoshiro_initial_, sizeof(uint4) * width * height, cudaMemcpyDeviceToDevice));
}

void CudaRenderer::refresh_camera()
{
    update_camera<<<1, 1>>>(camera_, *render_info_);
    CCE(cudaGetLastError());
    CCE(cudaDeviceSynchronize());
}

void CudaRenderer::refresh_texture(const int32_t index) const
{
    const TextureInfo* texture = world_info_->textures_[index];

    if (texture->type == SOLID)
		CCE(cudaMemcpy(host_texture_data_[index], texture, sizeof(SolidInfo), cudaMemcpyHostToDevice));
    else if (texture->type == IMAGE)
		CCE(cudaMemcpy(host_texture_data_[index], texture, sizeof(ImageInfo), cudaMemcpyHostToDevice));
    else if (texture->type == CHECKER)
		CCE(cudaMemcpy(host_texture_data_[index], texture, sizeof(CheckerInfo), cudaMemcpyHostToDevice));

    CCE(cudaMemcpy(device_texture_data_ + index, host_texture_data_ + index, sizeof(TextureInfo*), cudaMemcpyHostToDevice));

    update_texture<<<1, 1>>>(world_, index, device_texture_data_);
    CCE(cudaGetLastError());
    CCE(cudaDeviceSynchronize());
}

void CudaRenderer::refresh_material(const int32_t index) const
{
	const MaterialInfo* material = world_info_->materials_[index];

    if (material->type == DIFFUSE)
		CCE(cudaMemcpy(host_material_data_[index], material, sizeof(DiffuseInfo), cudaMemcpyHostToDevice));
    else if (material->type == SPECULAR)
		CCE(cudaMemcpy(host_material_data_[index], material, sizeof(SpecularInfo), cudaMemcpyHostToDevice));
    else if (material->type == REFRACTIVE)
		CCE(cudaMemcpy(host_material_data_[index], material, sizeof(RefractiveInfo), cudaMemcpyHostToDevice));
	else if (material->type == ISOTROPIC)
		CCE(cudaMemcpy(host_material_data_[index], material, sizeof(IsotropicInfo), cudaMemcpyHostToDevice));

    CCE(cudaMemcpy(device_material_data_ + index, host_material_data_ + index, sizeof(MaterialInfo*), cudaMemcpyHostToDevice));

    update_material<<<1, 1>>>(world_, index, device_material_data_);
    CCE(cudaGetLastError());
    CCE(cudaDeviceSynchronize());
}

void CudaRenderer::refresh_object(const int32_t index) const
{
	const ObjectInfo* object = world_info_->objects_[index];

    if (object->type == SPHERE)
		CCE(cudaMemcpy(host_object_data_[index], object, sizeof(SphereInfo), cudaMemcpyHostToDevice));
	else if (object->type == PLANE)
		CCE(cudaMemcpy(host_object_data_[index], object, sizeof(PlaneInfo), cudaMemcpyHostToDevice));
	else if (object->type == CYLINDER)
		CCE(cudaMemcpy(host_object_data_[index], object, sizeof(CylinderInfo), cudaMemcpyHostToDevice));
	else if (object->type == CONE)
		CCE(cudaMemcpy(host_object_data_[index], object, sizeof(ConeInfo), cudaMemcpyHostToDevice));
    else if (object->type == MODEL)
	    CCE(cudaMemcpy(host_object_data_[index], object, sizeof(ModelInfo), cudaMemcpyHostToDevice));

    CCE(cudaMemcpy(device_object_data_ + index, host_object_data_ + index, sizeof(ObjectInfo*), cudaMemcpyHostToDevice));

    update_object<<<1, 1>>>(world_, index, device_object_data_);
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

    CCE(cudaFree(xoshiro_state_));
    CCE(cudaFree(xoshiro_initial_));
    CCE(cudaFree(accumulation_buffer_));
    CCE(cudaMalloc(reinterpret_cast<void**>(&accumulation_buffer_), sizeof(float4) * width * height));
    CCE(cudaMalloc(reinterpret_cast<void**>(&xoshiro_initial_), sizeof(uint4) * width * height));
    CCE(cudaMalloc(reinterpret_cast<void**>(&xoshiro_state_), sizeof(uint4) * width * height));

    random_init<<<blocks_, threads_>>>(width, height, xoshiro_initial_);
    CCE(cudaGetLastError());
    CCE(cudaDeviceSynchronize());
}

void CudaRenderer::recreate_sky()
{
    CCE(cudaFree(sky_info_->usable_hdr_data));

    if (sky_info_->buffered_hdr_data)
    {
        const uint64_t hdr_size = sizeof(float3) * sky_info_->hdr_width * sky_info_->hdr_height;
	    CCE(cudaMalloc(reinterpret_cast<void**>(&sky_info_->usable_hdr_data), hdr_size));
    	CCE(cudaMemcpy(sky_info_->usable_hdr_data, sky_info_->buffered_hdr_data, hdr_size, cudaMemcpyHostToDevice));
    }
    else
        sky_info_->usable_hdr_data = nullptr;
}

void CudaRenderer::map_frame_memory()
{
    fetch_frame_buffer();

	CCE(cudaMemcpy(render_info_->frame_data, frame_buffer_, render_info_->frame_size, cudaMemcpyDeviceToHost));

    CCE(cudaFree(frame_buffer_));
}

void CudaRenderer::allocate_world()
{
    const std::vector<TextureInfo*> texture_data = world_info_->textures_;
	const std::vector<MaterialInfo*> material_data = world_info_->materials_;
	const std::vector<ObjectInfo*> object_data = world_info_->objects_;
    const auto texture_count = texture_data.size();
    const auto material_count = material_data.size();
    const auto object_count = object_data.size();
    host_texture_data_ = new TextureInfo*[texture_count];
    host_material_data_ = new MaterialInfo*[material_count];
    host_object_data_ = new ObjectInfo*[object_count];

    for (uint64_t i = 0; i < texture_count; i++)
    {
        if (texture_data[i]->type == SOLID)
        {
	        CCE(cudaMalloc(reinterpret_cast<void**>(&host_texture_data_[i]), sizeof(SolidInfo)));
			CCE(cudaMemcpy(host_texture_data_[i], texture_data[i], sizeof(SolidInfo), cudaMemcpyHostToDevice));
        }
    	else if (texture_data[i]->type == IMAGE)
        {
            const auto image_data = dynamic_cast<ImageInfo*>(texture_data[i]);
            const uint64_t image_size = sizeof(float) * image_data->width * image_data->height * 3;

            CCE(cudaMalloc(reinterpret_cast<void**>(&image_data->usable_data), image_size));
			CCE(cudaMemcpy(image_data->usable_data, image_data->buffered_data, image_size, cudaMemcpyHostToDevice));

	        CCE(cudaMalloc(reinterpret_cast<void**>(&host_texture_data_[i]), sizeof(ImageInfo)));
			CCE(cudaMemcpy(host_texture_data_[i], image_data, sizeof(ImageInfo), cudaMemcpyHostToDevice));
        }
        else if (texture_data[i]->type == CHECKER)
        {
	        CCE(cudaMalloc(reinterpret_cast<void**>(&host_texture_data_[i]), sizeof(CheckerInfo)));
			CCE(cudaMemcpy(host_texture_data_[i], texture_data[i], sizeof(CheckerInfo), cudaMemcpyHostToDevice));
        }
    }

    for (uint64_t i = 0; i < material_count; i++)
    {
        if (material_data[i]->type == DIFFUSE)
        {
	        CCE(cudaMalloc(reinterpret_cast<void**>(&host_material_data_[i]), sizeof(DiffuseInfo)));
			CCE(cudaMemcpy(host_material_data_[i], material_data[i], sizeof(DiffuseInfo), cudaMemcpyHostToDevice));
        }
        else if (material_data[i]->type == SPECULAR)
        {
	        CCE(cudaMalloc(reinterpret_cast<void**>(&host_material_data_[i]), sizeof(SpecularInfo)));
			CCE(cudaMemcpy(host_material_data_[i], material_data[i], sizeof(SpecularInfo), cudaMemcpyHostToDevice));
        }
        else if (material_data[i]->type == REFRACTIVE)
        {
	        CCE(cudaMalloc(reinterpret_cast<void**>(&host_material_data_[i]), sizeof(RefractiveInfo)));
			CCE(cudaMemcpy(host_material_data_[i], material_data[i], sizeof(RefractiveInfo), cudaMemcpyHostToDevice));
        }
    	else if (material_data[i]->type == ISOTROPIC)
        {
	        CCE(cudaMalloc(reinterpret_cast<void**>(&host_material_data_[i]), sizeof(IsotropicInfo)));
			CCE(cudaMemcpy(host_material_data_[i], material_data[i], sizeof(IsotropicInfo), cudaMemcpyHostToDevice));
        }
    }

    for (uint64_t i = 0; i < object_count; i++)
    {
        if (object_data[i]->type == SPHERE)
        {
	        CCE(cudaMalloc(reinterpret_cast<void**>(&host_object_data_[i]), sizeof(SphereInfo)));
			CCE(cudaMemcpy(host_object_data_[i], object_data[i], sizeof(SphereInfo), cudaMemcpyHostToDevice));
        }
    	else if (object_data[i]->type == PLANE)
        {
	        CCE(cudaMalloc(reinterpret_cast<void**>(&host_object_data_[i]), sizeof(PlaneInfo)));
			CCE(cudaMemcpy(host_object_data_[i], object_data[i], sizeof(PlaneInfo), cudaMemcpyHostToDevice));
        }
    	else if (object_data[i]->type == CYLINDER)
        {
	        CCE(cudaMalloc(reinterpret_cast<void**>(&host_object_data_[i]), sizeof(CylinderInfo)));
			CCE(cudaMemcpy(host_object_data_[i], object_data[i], sizeof(CylinderInfo), cudaMemcpyHostToDevice));
        }
    	else if (object_data[i]->type == CONE)
        {
	        CCE(cudaMalloc(reinterpret_cast<void**>(&host_object_data_[i]), sizeof(ConeInfo)));
			CCE(cudaMemcpy(host_object_data_[i], object_data[i], sizeof(ConeInfo), cudaMemcpyHostToDevice));
        }
        else if (object_data[i]->type == MODEL)
        {
	        const auto model_data = dynamic_cast<ModelInfo*>(object_data[i]);

            CCE(cudaMalloc(reinterpret_cast<void**>(&model_data->usable_vertices), 3 * model_data->triangle_count * sizeof(Vertex)));
			CCE(cudaMemcpy(model_data->usable_vertices, model_data->buffered_vertices, 3 * model_data->triangle_count * sizeof(Vertex), cudaMemcpyHostToDevice));

            CCE(cudaMalloc(reinterpret_cast<void**>(&host_object_data_[i]), sizeof(ModelInfo)));
			CCE(cudaMemcpy(host_object_data_[i], model_data, sizeof(ModelInfo), cudaMemcpyHostToDevice));
        }
    }

    CCE(cudaMalloc(reinterpret_cast<void**>(&device_texture_data_), texture_count * sizeof(TextureInfo*)));
    CCE(cudaMalloc(reinterpret_cast<void**>(&device_material_data_), material_count * sizeof(MaterialInfo*)));
    CCE(cudaMalloc(reinterpret_cast<void**>(&device_object_data_), object_count * sizeof(ObjectInfo*)));
	CCE(cudaMemcpy(device_texture_data_, host_texture_data_, texture_count * sizeof(TextureInfo*), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(device_material_data_, host_material_data_, material_count * sizeof(MaterialInfo*), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(device_object_data_, host_object_data_, object_count * sizeof(ObjectInfo*), cudaMemcpyHostToDevice));

    CCE(cudaMalloc(reinterpret_cast<void**>(&world_), sizeof(World*)));
    create_world<<<1, 1>>>(
        device_object_data_, 
        device_material_data_, 
        device_texture_data_, 
        static_cast<int32_t>(object_count), 
        static_cast<int32_t>(material_count), 
        static_cast<int32_t>(texture_count), world_);

	CCE(cudaGetLastError());
    CCE(cudaDeviceSynchronize());
}

void CudaRenderer::deallocate_world() const
{
    CCE(cudaDeviceSynchronize());
	delete_world<<<1, 1>>>(world_);
    CCE(cudaGetLastError());
    CCE(cudaFree(world_));

    CCE(cudaFree(device_object_data_));
    CCE(cudaFree(device_material_data_));
    CCE(cudaFree(device_texture_data_));

    for (uint64_t i = 0; i < world_info_->objects_.size(); i++)
    {
        if (world_info_->objects_[i]->type == MODEL)
            CCE(cudaFree(dynamic_cast<ModelInfo*>(world_info_->objects_[i])->usable_vertices));

	    CCE(cudaFree(host_object_data_[i]));
    }
    
    for (uint64_t i = 0; i < world_info_->materials_.size(); i++)
    {
	    CCE(cudaFree(host_material_data_[i]));
    }

	for (uint64_t i = 0; i < world_info_->textures_.size(); i++)
    {
        if (world_info_->textures_[i]->type == IMAGE)
            CCE(cudaFree(dynamic_cast<ImageInfo*>(world_info_->textures_[i])->usable_data));

	    CCE(cudaFree(host_texture_data_[i]));
    }

    delete[] host_object_data_;
    delete[] host_material_data_;
    delete[] host_texture_data_;
}

void CudaRenderer::fetch_frame_buffer()
{
    cudaExternalMemoryHandleDesc handle_desc = {};

#if defined(_WIN32)
    handle_desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    handle_desc.handle.win32.handle = reinterpret_cast<void*>(render_info_->frame_handle);
#elif defined(__linux__) || defined(__APPLE__)
    handle_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    handle_desc.handle.fd = render_info_->frame_handle;
#endif

    handle_desc.size = render_info_->frame_size;

    cudaExternalMemory_t external_memory = {};
    CCE(cudaImportExternalMemory(&external_memory, &handle_desc));

    cudaExternalMemoryBufferDesc buffer_desc = {};
    buffer_desc.size = render_info_->frame_size;
    buffer_desc.offset = 0;

    CCE(cudaExternalMemoryGetMappedBuffer(reinterpret_cast<void**>(&frame_buffer_), external_memory, &buffer_desc));
}
