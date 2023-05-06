#include "CpuRenderer.hpp"

#include <cstring>

CpuRenderer::CpuRenderer(const RenderInfo* render_info, const WorldInfo* world_info, SkyInfo* sky_info)
	: render_info_(render_info), world_info_(world_info), sky_info_(sky_info)
{
	const uint64_t image_size = static_cast<uint64_t>(render_info_->width) * render_info_->height;

	accumulation_buffer_ = new float4[image_size];
	xoshiro_initial_ = new uint4[image_size];
	xoshiro_state_ = new uint4[image_size];

	random_init();
	random_refresh();
	allocate_world();

	camera_ = new Camera(
			render_info_->camera_position,
			render_info_->camera_direction,
			render_info_->fov,
			static_cast<float>(render_info_->width) / static_cast<float>(render_info_->height),
			render_info_->aperture,
			render_info_->focus_distance);

	if (sky_info_->buffered_hdr_data)
		sky_info_->usable_hdr_data = reinterpret_cast<float3*>(sky_info_->buffered_hdr_data);
}

CpuRenderer::~CpuRenderer()
{
	delete camera_;

	deallocate_world();

	delete[] xoshiro_state_;
	delete[] xoshiro_initial_;
	delete[] accumulation_buffer_;
}

void CpuRenderer::render()
{
	const auto width = static_cast<int32_t>(render_info_->width);
	const auto height = static_cast<int32_t>(render_info_->height);
	float* frame_data = render_info_->frame_data;

	if (render_info_->render_mode == PROGRESSIVE)
	{
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
				const Ray ray = camera_->cast_ray(&local_random_state, u, v);
				const float3 color = sqrt(calculate_color(ray, &world_, *sky_info_, render_info_->max_depth, &local_random_state));

				accumulation_buffer_[pixel_index] += make_float4(color, 1.0f);
				frame_data[pixel_index << 2] = accumulation_buffer_[pixel_index].x / static_cast<float>(render_info_->frames_since_refresh);
				frame_data[(pixel_index << 2) + 1] = accumulation_buffer_[pixel_index].y / static_cast<float>(render_info_->frames_since_refresh);
				frame_data[(pixel_index << 2) + 2] = accumulation_buffer_[pixel_index].z / static_cast<float>(render_info_->frames_since_refresh);
				frame_data[(pixel_index << 2) + 3] = accumulation_buffer_[pixel_index].w / static_cast<float>(render_info_->frames_since_refresh);
			}
		}
	}
	else if (render_info_->render_mode == STATIC)
	{
		memset(frame_data, 0, render_info_->frame_size);

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
					const Ray ray = camera_->cast_ray(&local_random_state, u, v);
					const float3 color = sqrt(calculate_color(ray, &world_, *sky_info_, render_info_->max_depth, &local_random_state));

					frame_data[pixel_index << 2] += color.x / static_cast<float>(render_info_->samples_per_pixel);
					frame_data[(pixel_index << 2) + 1] += color.y / static_cast<float>(render_info_->samples_per_pixel);
					frame_data[(pixel_index << 2) + 2] += color.z / static_cast<float>(render_info_->samples_per_pixel);
					frame_data[(pixel_index << 2) + 3] += 1.0f / static_cast<float>(render_info_->samples_per_pixel);
				}
			}
		}
	}
}

void CpuRenderer::refresh_buffer()
{
	const uint32_t width = render_info_->width;
	const uint32_t height = render_info_->height;

	memset(accumulation_buffer_, 0, sizeof(float4) * width * height);
	random_refresh();
}

void CpuRenderer::refresh_camera()
{
	camera_->update(
		render_info_->camera_position,
		render_info_->camera_direction,
		render_info_->fov,
		render_info_->aperture,
		render_info_->focus_distance);
}

void CpuRenderer::refresh_object(const int32_t index) const
{
	world_->update_object(index, world_info_->objects_[index]);
}

void CpuRenderer::refresh_material(const int32_t index) const
{
	world_->update_material(index, world_info_->materials_[index]);
}

void CpuRenderer::refresh_texture(const int32_t index) const
{
	world_->update_texture(index, world_info_->textures_[index]);
}

void CpuRenderer::recreate_camera()
{
	delete camera_;
	camera_ = new Camera(
			render_info_->camera_position,
			render_info_->camera_direction,
			render_info_->fov,
			static_cast<float>(render_info_->width) / static_cast<float>(render_info_->height),
			render_info_->aperture,
			render_info_->focus_distance);
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
	if (sky_info_->buffered_hdr_data)
		sky_info_->usable_hdr_data = reinterpret_cast<float3*>(sky_info_->buffered_hdr_data);
	else
		sky_info_->usable_hdr_data = nullptr;
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
	const auto texture_data = world_info_->textures_;
	const auto material_data = world_info_->materials_;
    const auto object_data = world_info_->objects_;
	const auto texture_count = texture_data.size();
	const auto material_count = material_data.size();
	const auto object_count = object_data.size();

	texture_data_ = const_cast<TextureInfo**>(texture_data.data());
	material_data_ = const_cast<MaterialInfo**>(material_data.data());
	object_data_ = const_cast<ObjectInfo**>(object_data.data());

	for (int32_t i = 0; i < static_cast<int32_t>(texture_count); i++)
	{
		if (texture_data_[i]->type == IMAGE)
		{
			const auto image_data = dynamic_cast<ImageInfo*>(texture_data_[i]);
			image_data->usable_data = image_data->buffered_data;
		}
	}

	for (int32_t i = 0; i < static_cast<int32_t>(object_count); i++)
	{
		if (object_data_[i]->type == MODEL)
		{
			const auto model_data = dynamic_cast<ModelInfo*>(object_data_[i]);
			model_data->usable_vertices = model_data->buffered_vertices;
		}
	}

	world_ = new World(
		object_data_, 
		material_data_,
		texture_data_, 
		static_cast<int32_t>(object_count), 
		static_cast<int32_t>(material_count), 
		static_cast<int32_t>(texture_count));
}

void CpuRenderer::deallocate_world() const
{
	delete world_;
}