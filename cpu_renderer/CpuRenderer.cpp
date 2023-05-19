#include "stdafx.h"
#include "CpuRenderer.hpp"

CpuRenderer::CpuRenderer(const RenderInfo* render_info, const WorldInfo* world_info, const SkyInfo* sky_info, const CameraInfo* camera_info)
	: render_info_(render_info), world_info_(world_info), sky_info_(sky_info), camera_info_(camera_info)
{
	const uint64_t image_size = static_cast<uint64_t>(render_info_->width) * render_info_->height;

	accumulation_buffer_ = new float4[image_size];
	xoshiro_initial_ = new uint4[image_size];
	xoshiro_state_ = new uint4[image_size];

	random_init();
	random_refresh();
	allocate_world();

	if (sky_info_->h_hdr_data)
		sky_info_->d_hdr_data = reinterpret_cast<float3*>(sky_info_->h_hdr_data);
}

CpuRenderer::~CpuRenderer()
{
	deallocate_world();

	delete[] xoshiro_state_;
	delete[] xoshiro_initial_;
	delete[] accumulation_buffer_;
}

void CpuRenderer::render_static() const
{
	const auto width = static_cast<int32_t>(render_info_->width);
	const auto height = static_cast<int32_t>(render_info_->height);
	float* frame_data = render_info_->frame_data;

#ifndef _DEBUG
#pragma omp parallel for schedule(dynamic)
#endif
	for (int32_t y = 0; y < static_cast<int32_t>(height); y++)
	{
		for (int32_t x = 0; x < static_cast<int32_t>(width); x++)
		{
			for (int32_t i = 0; i < render_info_->samples_per_pixel; i++)
			{
				const int32_t pixel_index = y * width + x;
				uint32_t local_random_state = xoshiro(&xoshiro_state_[y * width + x]);

				const float u = (static_cast<float>(x) + pcg(&local_random_state)) / static_cast<float>(width);
				const float v = (static_cast<float>(y) + pcg(&local_random_state)) / static_cast<float>(height);
				const Ray ray = cast_ray(&local_random_state, u, v, *camera_info_);
				const float3 color = sqrt(calculate_color(ray, world_, *sky_info_, render_info_->max_depth, &local_random_state));

				frame_data[pixel_index << 2] += color.x / static_cast<float>(render_info_->samples_per_pixel);
				frame_data[(pixel_index << 2) + 1] += color.y / static_cast<float>(render_info_->samples_per_pixel);
				frame_data[(pixel_index << 2) + 2] += color.z / static_cast<float>(render_info_->samples_per_pixel);
				frame_data[(pixel_index << 2) + 3] += 1.0f / static_cast<float>(render_info_->samples_per_pixel);
			}
		}
	}
}

void CpuRenderer::render_progressive() const
{
	const auto width = static_cast<int32_t>(render_info_->width);
	const auto height = static_cast<int32_t>(render_info_->height);
	float* frame_data = render_info_->frame_data;

#ifndef _DEBUG
#pragma omp parallel for schedule(dynamic)
#endif
	for (int32_t y = 0; y < height; y++)
	{
		for (int32_t x = 0; x < width; x++)
		{
			const int32_t pixel_index = y * width + x;
			uint32_t local_random_state = xoshiro(&xoshiro_state_[y * width + x]);

			const float u = (static_cast<float>(x) + pcg(&local_random_state)) / static_cast<float>(width);
			const float v = (static_cast<float>(y) + pcg(&local_random_state)) / static_cast<float>(height);
			const Ray ray = cast_ray(&local_random_state, u, v, *camera_info_);
			const float3 color = sqrt(calculate_color(ray, world_, *sky_info_, render_info_->max_depth, &local_random_state));

			accumulation_buffer_[pixel_index] += make_float4(color, 1.0f);
			frame_data[pixel_index << 2] = accumulation_buffer_[pixel_index].x / static_cast<float>(render_info_->frames_since_refresh);
			frame_data[(pixel_index << 2) + 1] = accumulation_buffer_[pixel_index].y / static_cast<float>(render_info_->frames_since_refresh);
			frame_data[(pixel_index << 2) + 2] = accumulation_buffer_[pixel_index].z / static_cast<float>(render_info_->frames_since_refresh);
			frame_data[(pixel_index << 2) + 3] = accumulation_buffer_[pixel_index].w / static_cast<float>(render_info_->frames_since_refresh);
		}
	}
}

void CpuRenderer::render()
{
	if (render_info_->progressive)
	{
		render_progressive();
	}
	else
	{
		memset(render_info_->frame_data, 0, render_info_->frame_size);
		render_static();
	}
}

void CpuRenderer::refresh_buffer()
{
	const uint32_t width = render_info_->width;
	const uint32_t height = render_info_->height;

	memset(accumulation_buffer_, 0, sizeof(float4) * width * height);
	random_refresh();
}

void CpuRenderer::recreate_image()
{
	const uint64_t image_size = static_cast<uint64_t>(render_info_->width) * render_info_->height;

	delete[] xoshiro_state_;
	delete[] xoshiro_initial_;
	delete[] accumulation_buffer_;
	accumulation_buffer_ = new float4[image_size];
	xoshiro_initial_ = new uint4[image_size];
	xoshiro_state_ = new uint4[image_size];

	random_init();
	random_refresh();
}

void CpuRenderer::recreate_sky()
{
	if (sky_info_->h_hdr_data)
		sky_info_->d_hdr_data = reinterpret_cast<float3*>(sky_info_->h_hdr_data);
	else
		sky_info_->d_hdr_data = nullptr;
}

void CpuRenderer::random_init() const
{
	for (int32_t y = 0; y < static_cast<int32_t>(render_info_->height); y++)
	{
		for (int32_t x = 0; x < static_cast<int32_t>(render_info_->width); x++)
		{
			const int32_t pixel_index = y * static_cast<int32_t>(render_info_->width) + x;
			xoshiro_initial_[pixel_index] = make_uint4(
				pixel_index + 15072003,
				pixel_index + 15112001,
				pixel_index + 10021151,
				pixel_index + 30027051);
		}
	}
}

void CpuRenderer::random_refresh() const
{
	const uint64_t xoshiro_size = sizeof(uint4) * render_info_->width * render_info_->height;
	memcpy_s(xoshiro_state_, xoshiro_size, xoshiro_initial_, xoshiro_size);
}

void CpuRenderer::allocate_world()
{
	auto& textures = world_info_->textures_;
	auto& materials = world_info_->materials_;
	auto& objects = world_info_->objects_;
	const auto texture_count = textures.size();
	const auto material_count = materials.size();
	const auto object_count = objects.size();

	const auto& texture_data = const_cast<Texture*>(textures.data());
	const auto& material_data = const_cast<Material*>(materials.data());
	const auto& object_data = const_cast<Object*>(objects.data());

	for (uint64_t i = 0; i < texture_count; i++)
	{
		if (texture_data[i].type == TextureType::IMAGE)
		{
			const auto image_data = &texture_data[i].image;
			image_data->d_data = image_data->h_data;
		}
	}

	for (uint64_t i = 0; i < object_count; i++)
	{
		if (object_data[i].type == ObjectType::MODEL)
		{
			const auto model_data = &object_data[i].model;
			model_data->d_vertices = model_data->h_vertices;
			model_data->d_indices = model_data->h_indices;
		}
	}

	world_ = new World(
		object_data,
		material_data,
		texture_data,
		static_cast<int32_t>(object_count),
		static_cast<int32_t>(material_count),
		static_cast<int32_t>(texture_count));
}

void CpuRenderer::deallocate_world() const
{
	delete world_;
}