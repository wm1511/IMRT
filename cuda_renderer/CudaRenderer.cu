// ReSharper disable once CppPrecompiledHeaderIsNotIncluded
#include "CudaRenderer.cuh"
#include "CudaKernels.cuh"

#include "../common/Utils.hpp"

CudaRenderer::CudaRenderer(const RenderInfo* render_info, WorldInfo* world_info, const SkyInfo* sky_info, const CameraInfo* camera_info)
	: render_info_(render_info), world_info_(world_info), sky_info_(sky_info), camera_info_(camera_info)
{
	const uint32_t width = render_info_->width;
	const uint32_t height = render_info_->height;
	constexpr int32_t thread_x = 16;
	constexpr int32_t thread_y = 16;
	blocks_ = dim3((width + thread_x - 1) / thread_x, (height + thread_y - 1) / thread_y);
	threads_ = dim3(thread_x, thread_y);

	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	CCE(cudaMalloc(reinterpret_cast<void**>(&accumulation_buffer_), sizeof(float4) * width * height));
	CCE(cudaMalloc(reinterpret_cast<void**>(&xoshiro_initial_), sizeof(uint4) * width * height));
	CCE(cudaMalloc(reinterpret_cast<void**>(&xoshiro_state_), sizeof(uint4) * width * height));

	random_init<<<blocks_, threads_>>>(width, height, xoshiro_initial_);
	CCE(cudaGetLastError());
	CCE(cudaDeviceSynchronize());

	CCE(cudaMemcpy(xoshiro_state_, xoshiro_initial_, sizeof(uint4) * width * height, cudaMemcpyDeviceToDevice));

	allocate_world();

	if (sky_info_->h_hdr_data)
	{
		const uint64_t hdr_size = sizeof(float3) * sky_info_->hdr_width * sky_info_->hdr_height;
		CCE(cudaMalloc(reinterpret_cast<void**>(&sky_info_->d_hdr_data), hdr_size));
		CCE(cudaMemcpy(sky_info_->d_hdr_data, sky_info_->h_hdr_data, hdr_size, cudaMemcpyHostToDevice));
	}
}

CudaRenderer::~CudaRenderer()
{
	if (sky_info_->h_hdr_data)
		CCE(cudaFree(sky_info_->d_hdr_data));

	deallocate_world();

	CCE(cudaFree(xoshiro_state_));
	CCE(cudaFree(xoshiro_initial_));
	CCE(cudaFree(accumulation_buffer_));
	cudaDeviceReset();
}

void CudaRenderer::render()
{
	const auto frame_buffer = static_cast<float4*>(fetch_external_memory(render_info_->frame_handle, render_info_->frame_size));

	if (render_info_->progressive)
	{
		render_pixel_progressive<<<blocks_, threads_>>>(frame_buffer, accumulation_buffer_, world_, *sky_info_, *render_info_, *camera_info_, xoshiro_state_);
	}
	else
	{
		CCE(cudaMemset(frame_buffer, 0, render_info_->frame_size));

		for (int32_t i = 0; i < render_info_->samples_per_pixel; i++)
			render_pixel_static<<<blocks_, threads_>>>(frame_buffer, world_, *sky_info_, *render_info_, *camera_info_, xoshiro_state_);
	}

	CCE(cudaGetLastError());
	CCE(cudaDeviceSynchronize());

	CCE(cudaFree(frame_buffer));
}

void CudaRenderer::refresh_buffer()
{
	const uint32_t width = render_info_->width;
	const uint32_t height = render_info_->height;

	CCE(cudaMemset(accumulation_buffer_, 0, sizeof(float4) * width * height));
	CCE(cudaMemcpy(xoshiro_state_, xoshiro_initial_, sizeof(uint4) * width * height, cudaMemcpyDeviceToDevice));
}

void CudaRenderer::refresh_texture(const int32_t index) const
{
	CCE(cudaMemcpy(d_texture_data_ + index, &world_info_->textures_[index], sizeof(Texture), cudaMemcpyHostToDevice));
}

void CudaRenderer::refresh_material(const int32_t index) const
{
	CCE(cudaMemcpy(d_material_data_ + index, &world_info_->materials_[index], sizeof(Material), cudaMemcpyHostToDevice));
}

void CudaRenderer::refresh_object(const int32_t index) const
{
	CCE(cudaMemcpy(d_object_data_ + index, &world_info_->objects_[index], sizeof(Object), cudaMemcpyHostToDevice));
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
	CCE(cudaFree(sky_info_->d_hdr_data));

	if (sky_info_->h_hdr_data)
	{
		const uint64_t hdr_size = sizeof(float3) * sky_info_->hdr_width * sky_info_->hdr_height;
		CCE(cudaMalloc(reinterpret_cast<void**>(&sky_info_->d_hdr_data), hdr_size));
		CCE(cudaMemcpy(sky_info_->d_hdr_data, sky_info_->h_hdr_data, hdr_size, cudaMemcpyHostToDevice));
	}
	else
		sky_info_->d_hdr_data = nullptr;
}

void CudaRenderer::map_frame_memory()
{
	const auto frame_buffer = static_cast<float4*>(fetch_external_memory(render_info_->frame_handle, render_info_->frame_size));

	CCE(cudaMemcpy(render_info_->frame_data, frame_buffer, render_info_->frame_size, cudaMemcpyDeviceToHost));
	CCE(cudaFree(frame_buffer));
}

void CudaRenderer::allocate_world()
{
	auto& textures = world_info_->textures_;
	const auto& materials = world_info_->materials_;
	auto& objects = world_info_->objects_;

	for (auto& texture : textures)
	{
		if (texture.type == TextureType::IMAGE)
		{
			const auto image_data = &texture.image;
			const uint64_t image_size = sizeof(float) * image_data->width * image_data->height * 3;

			CCE(cudaMalloc(reinterpret_cast<void**>(&image_data->d_data), image_size));
			CCE(cudaMemcpy(image_data->d_data, image_data->h_data, image_size, cudaMemcpyHostToDevice));
		}
	}

	for (auto& object : objects)
	{
		if (object.type == ObjectType::MODEL)
		{
			const auto model_data = &object.model;

			CCE(cudaMalloc(reinterpret_cast<void**>(&model_data->d_vertices), model_data->vertex_count * sizeof(float3)));
			CCE(cudaMemcpy(model_data->d_vertices, model_data->h_vertices, model_data->vertex_count * sizeof(float3), cudaMemcpyHostToDevice));

			CCE(cudaMalloc(reinterpret_cast<void**>(&model_data->d_indices), model_data->index_count * sizeof(uint3)));
			CCE(cudaMemcpy(model_data->d_indices, model_data->h_indices, model_data->index_count * sizeof(uint3), cudaMemcpyHostToDevice));

			CCE(cudaMalloc(reinterpret_cast<void**>(&model_data->d_normals), model_data->vertex_count * sizeof(float3)));
			CCE(cudaMemcpy(model_data->d_normals, model_data->h_normals, model_data->vertex_count * sizeof(float3), cudaMemcpyHostToDevice));

			CCE(cudaMalloc(reinterpret_cast<void**>(&model_data->d_uv), model_data->vertex_count * sizeof(float2)));
			CCE(cudaMemcpy(model_data->d_uv, model_data->h_uv, model_data->vertex_count * sizeof(float2), cudaMemcpyHostToDevice));
		}
	}

	CCE(cudaMalloc(reinterpret_cast<void**>(&d_texture_data_), textures.size() * sizeof(Texture)));
	CCE(cudaMalloc(reinterpret_cast<void**>(&d_material_data_), materials.size() * sizeof(Material)));
	CCE(cudaMalloc(reinterpret_cast<void**>(&d_object_data_), objects.size() * sizeof(Object)));
	CCE(cudaMemcpy(d_texture_data_, textures.data(), textures.size() * sizeof(Texture), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_material_data_, materials.data(), materials.size() * sizeof(Material), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_object_data_, objects.data(), objects.size() * sizeof(Object), cudaMemcpyHostToDevice));

	const auto world = World(
		d_object_data_, 
		d_material_data_, 
		d_texture_data_, 
		static_cast<int32_t>(objects.size()));

	CCE(cudaMalloc(reinterpret_cast<void**>(&world_), sizeof(World)));
	CCE(cudaMemcpy(world_, &world, sizeof(World), cudaMemcpyHostToDevice));
}

void CudaRenderer::deallocate_world()
{
	CCE(cudaFree(world_));

	CCE(cudaFree(d_object_data_));
	CCE(cudaFree(d_material_data_));
	CCE(cudaFree(d_texture_data_));

	for (const auto& object : world_info_->objects_)
	{
		if (object.type == ObjectType::MODEL)
		{
			CCE(cudaFree(object.model.d_vertices));
			CCE(cudaFree(object.model.d_indices));
			CCE(cudaFree(object.model.d_normals));
			CCE(cudaFree(object.model.d_uv));
		}
	}

	for (const auto& texture : world_info_->textures_)
	{
		if (texture.type == TextureType::IMAGE)
			CCE(cudaFree(texture.image.d_data));
	}
}
