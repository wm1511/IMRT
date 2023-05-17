// ReSharper disable once CppPrecompiledHeaderIsNotIncluded
#include "CudaRenderer.cuh"
#include "CudaKernels.cuh"

#include "../common/Utils.hpp"

#include <cuda.h>

CudaRenderer::CudaRenderer(const RenderInfo* render_info, const WorldInfo* world_info, const SkyInfo* sky_info, const CameraInfo* camera_info)
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
	frame_buffer_ = static_cast<float4*>(fetch_external_memory(render_info_->frame_handle, render_info_->frame_size));

	if (render_info_->progressive)
	{
		render_pixel_progressive<<<blocks_, threads_>>>(frame_buffer_, accumulation_buffer_, world_, *sky_info_, *render_info_, *camera_info_, xoshiro_state_);
	}
	else
	{
		CCE(cudaMemset(frame_buffer_, 0, render_info_->frame_size));

		for (int32_t i = 0; i < render_info_->samples_per_pixel; i++)
			render_pixel_static<<<blocks_, threads_>>>(frame_buffer_, world_, *sky_info_, *render_info_, *camera_info_, xoshiro_state_);
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

void CudaRenderer::refresh_texture(const int32_t index) const
{
	const TextureInfo* texture = world_info_->textures_[index];

	CCE(cudaMemcpy(h_texture_data_[index], texture, texture->get_size(), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_texture_data_ + index, h_texture_data_ + index, sizeof(CUdeviceptr), cudaMemcpyHostToDevice));

	update_texture<<<1, 1>>>(world_, index, d_texture_data_);
	CCE(cudaGetLastError());
	CCE(cudaDeviceSynchronize());
}

void CudaRenderer::refresh_material(const int32_t index) const
{
	const MaterialInfo* material = world_info_->materials_[index];

	CCE(cudaMemcpy(h_material_data_[index], material, material->get_size(), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_material_data_ + index, h_material_data_ + index, sizeof(CUdeviceptr), cudaMemcpyHostToDevice));

	update_material<<<1, 1 >>>(world_, index, d_material_data_);
	CCE(cudaGetLastError());
	CCE(cudaDeviceSynchronize());
}

void CudaRenderer::refresh_object(const int32_t index) const
{
	const ObjectInfo* object = world_info_->objects_[index];

	CCE(cudaMemcpy(h_object_data_[index], object, object->get_size(), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_object_data_ + index, h_object_data_ + index, sizeof(CUdeviceptr), cudaMemcpyHostToDevice));

	update_object<<<1, 1 >>>(world_, index, d_object_data_);
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
	frame_buffer_ = static_cast<float4*>(fetch_external_memory(render_info_->frame_handle, render_info_->frame_size));

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
	h_texture_data_ = new TextureInfo * [texture_count];
	h_material_data_ = new MaterialInfo * [material_count];
	h_object_data_ = new ObjectInfo * [object_count];

	for (uint64_t i = 0; i < texture_count; i++)
	{
		if (texture_data[i]->type == TextureType::IMAGE)
		{
			const auto image_data = dynamic_cast<ImageInfo*>(texture_data[i]);
			const uint64_t image_size = sizeof(float) * image_data->width * image_data->height * 3;

			CCE(cudaMalloc(reinterpret_cast<void**>(&image_data->d_data), image_size));
			CCE(cudaMemcpy(image_data->d_data, image_data->h_data, image_size, cudaMemcpyHostToDevice));
		}

		CCE(cudaMalloc(reinterpret_cast<void**>(&h_texture_data_[i]), texture_data[i]->get_size()));
		CCE(cudaMemcpy(h_texture_data_[i], texture_data[i], texture_data[i]->get_size(), cudaMemcpyHostToDevice));
	}

	for (uint64_t i = 0; i < material_count; i++)
	{
		CCE(cudaMalloc(reinterpret_cast<void**>(&h_material_data_[i]), material_data[i]->get_size()));
		CCE(cudaMemcpy(h_material_data_[i], material_data[i], material_data[i]->get_size(), cudaMemcpyHostToDevice));
	}

	for (uint64_t i = 0; i < object_count; i++)
	{
		if (object_data[i]->type == ObjectType::MODEL)
		{
			const auto model_data = dynamic_cast<ModelInfo*>(object_data[i]);

			CCE(cudaMalloc(reinterpret_cast<void**>(&model_data->d_vertices), model_data->vertex_count * sizeof(Vertex)));
			CCE(cudaMemcpy(model_data->d_vertices, model_data->h_vertices, model_data->vertex_count * sizeof(Vertex), cudaMemcpyHostToDevice));

			CCE(cudaMalloc(reinterpret_cast<void**>(&model_data->d_indices), model_data->index_count * sizeof(uint32_t)));
			CCE(cudaMemcpy(model_data->d_indices, model_data->h_indices, model_data->index_count * sizeof(uint32_t), cudaMemcpyHostToDevice));
		}

		CCE(cudaMalloc(reinterpret_cast<void**>(&h_object_data_[i]), object_data[i]->get_size()));
		CCE(cudaMemcpy(h_object_data_[i], object_data[i], object_data[i]->get_size(), cudaMemcpyHostToDevice));
	}

	CCE(cudaMalloc(reinterpret_cast<void**>(&d_texture_data_), texture_count * sizeof(CUdeviceptr)));
	CCE(cudaMalloc(reinterpret_cast<void**>(&d_material_data_), material_count * sizeof(CUdeviceptr)));
	CCE(cudaMalloc(reinterpret_cast<void**>(&d_object_data_), object_count * sizeof(CUdeviceptr)));
	CCE(cudaMemcpy(d_texture_data_, h_texture_data_, texture_count * sizeof(CUdeviceptr), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_material_data_, h_material_data_, material_count * sizeof(CUdeviceptr), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_object_data_, h_object_data_, object_count * sizeof(CUdeviceptr), cudaMemcpyHostToDevice));

	CCE(cudaMalloc(reinterpret_cast<void**>(&world_), sizeof(CUdeviceptr)));
	create_world<<<1, 1 >>>(
		d_object_data_,
		d_material_data_,
		d_texture_data_,
		static_cast<int32_t>(object_count),
		static_cast<int32_t>(material_count),
		static_cast<int32_t>(texture_count), world_);

	CCE(cudaGetLastError());
	CCE(cudaDeviceSynchronize());
}

void CudaRenderer::deallocate_world() const
{
	CCE(cudaDeviceSynchronize());
	delete_world << <1, 1 >> > (world_);
	CCE(cudaGetLastError());
	CCE(cudaFree(world_));

	CCE(cudaFree(d_object_data_));
	CCE(cudaFree(d_material_data_));
	CCE(cudaFree(d_texture_data_));

	for (uint64_t i = 0; i < world_info_->objects_.size(); i++)
	{
		if (world_info_->objects_[i]->type == ObjectType::MODEL)
		{
			CCE(cudaFree(dynamic_cast<ModelInfo*>(world_info_->objects_[i])->d_vertices));
			CCE(cudaFree(dynamic_cast<ModelInfo*>(world_info_->objects_[i])->d_indices));
		}

		CCE(cudaFree(h_object_data_[i]));
	}

	for (uint64_t i = 0; i < world_info_->materials_.size(); i++)
	{
		CCE(cudaFree(h_material_data_[i]));
	}

	for (uint64_t i = 0; i < world_info_->textures_.size(); i++)
	{
		if (world_info_->textures_[i]->type == TextureType::IMAGE)
			CCE(cudaFree(dynamic_cast<ImageInfo*>(world_info_->textures_[i])->d_data));

		CCE(cudaFree(h_texture_data_[i]));
	}

	delete[] h_object_data_;
	delete[] h_material_data_;
	delete[] h_texture_data_;
}
