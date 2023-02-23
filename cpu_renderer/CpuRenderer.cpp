#include "CpuRenderer.hpp"

#include <cstring>

CpuRenderer::CpuRenderer(const RenderInfo* render_info, const WorldInfo* world_info) : render_info_(render_info), world_info_(world_info)
{
	const uint64_t image_size = (uint64_t)render_info_->width * render_info_->height;

	accumulation_buffer_ = new float4[image_size];
	xoshiro_state_ = new uint4[image_size];

	random_init();
	allocate_world();

	camera_ = new Camera(
			render_info_->camera_position,
			render_info_->camera_direction,
			render_info_->fov,
			(float)render_info_->width / (float)render_info_->height,
			render_info_->aperture,
			render_info_->focus_distance);

	if (render_info_->hdr_data)
	{
		const uint64_t hdr_size = sizeof(float3) * render_info_->hdr_width * render_info_->hdr_height;
		hdr_data_ = new float3[hdr_size];
		memcpy_s(hdr_data_, hdr_size, render_info_->hdr_data, hdr_size);
	}
}

CpuRenderer::~CpuRenderer()
{
	delete[] hdr_data_;

	delete camera_;

	deallocate_world();

	delete[] xoshiro_state_;
	delete[] accumulation_buffer_;
}

void CpuRenderer::render(float* image_data)
{
	const int32_t width = (int32_t)render_info_->width;
	const int32_t height = (int32_t)render_info_->height;

	memset(image_data, 0, sizeof(float4) * width * height);

	if (render_info_->render_mode == PROGRESSIVE)
	{
#pragma omp parallel for schedule(dynamic)
		for (int32_t y = 0; y < height; y++)
		{
			for (int32_t x = 0; x < width; x++)
			{
				const int32_t pixel_index = y * width + x;
				uint32_t local_random_state = xoshiro(&xoshiro_state_[y * width + x]);

				const float u = ((float)x + pcg(&local_random_state)) / (float)width;
				const float v = ((float)y + pcg(&local_random_state)) / (float)height;
				const Ray ray = camera_->cast_ray(&local_random_state, u, v);
				const float3 color = sqrt(calculate_color(ray, &world_, hdr_data_, *render_info_, &local_random_state));

				accumulation_buffer_[pixel_index] += make_float4(color, 1.0f);
				image_data[pixel_index << 2] = accumulation_buffer_[pixel_index].x / (float)render_info_->frames_since_refresh;
				image_data[(pixel_index << 2) + 1] = accumulation_buffer_[pixel_index].y / (float)render_info_->frames_since_refresh;
				image_data[(pixel_index << 2) + 2] = accumulation_buffer_[pixel_index].z / (float)render_info_->frames_since_refresh;
				image_data[(pixel_index << 2) + 3] = accumulation_buffer_[pixel_index].w / (float)render_info_->frames_since_refresh;
			}
		}
	}
	else if (render_info_->render_mode == STATIC)
	{
#pragma omp parallel for schedule(dynamic)
		for (int32_t y = 0; y < static_cast<int32_t>(height); y++)
		{
			for (int32_t x = 0; x < static_cast<int32_t>(width); x++)
			{
				for (int32_t i = 0; i < render_info_->samples_per_pixel; i++)
				{
					const int32_t pixel_index = y * width + x;
					uint32_t local_random_state = xoshiro(&xoshiro_state_[y * width + x]);

					const float u = ((float)x + pcg(&local_random_state)) / (float)width;
					const float v = ((float)y + pcg(&local_random_state)) / (float)height;
					const Ray ray = camera_->cast_ray(&local_random_state, u, v);
					const float3 color = sqrt(calculate_color(ray, &world_, hdr_data_, *render_info_, &local_random_state));

					image_data[pixel_index << 2] += color.x / (float)render_info_->samples_per_pixel;
					image_data[(pixel_index << 2) + 1] += color.y / (float)render_info_->samples_per_pixel;
					image_data[(pixel_index << 2) + 2] += color.z / (float)render_info_->samples_per_pixel;
					image_data[(pixel_index << 2) + 3] += 1.0f / (float)render_info_->samples_per_pixel;
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
	random_init();
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

void CpuRenderer::recreate_camera()
{
	delete camera_;
	camera_ = new Camera(
			render_info_->camera_position,
			render_info_->camera_direction,
			render_info_->fov,
			(float)render_info_->width / (float)render_info_->height,
			render_info_->aperture,
			render_info_->focus_distance);
}

void CpuRenderer::recreate_image()
{
	const uint64_t image_size = (uint64_t)render_info_->width * render_info_->height;

	delete[] xoshiro_state_;
	delete[] accumulation_buffer_;
	accumulation_buffer_ = new float4[image_size];
	xoshiro_state_ = new uint4[image_size];
}

void CpuRenderer::recreate_sky()
{
	delete[] hdr_data_;

	if (render_info_->hdr_data)
	{
		const uint64_t hdr_size = sizeof(float3) * render_info_->hdr_width * render_info_->hdr_height;
		hdr_data_ = new float3[hdr_size];
		memcpy_s(hdr_data_, hdr_size, render_info_->hdr_data, hdr_size);
	}
	else
		hdr_data_ = nullptr;
}

void CpuRenderer::random_init() const
{
	for (int32_t y = 0; y < (int32_t)render_info_->height; y++)
	{
		for (int32_t x = 0; x < (int32_t)render_info_->width; x++)
		{
			const int32_t pixel_index = y * (int32_t)render_info_->width + x;
			xoshiro_state_[pixel_index] = make_uint4(
			   pixel_index + 15072003,
			   pixel_index + 15112001,
			   pixel_index + 10021151,
			   pixel_index + 30027051);
		}
	}
}

void CpuRenderer::allocate_world()
{
	const auto material_data = world_info_->materials_;
    const auto object_data = world_info_->objects_;
	const auto object_count = object_data.size();
	const auto material_count = material_data.size();

	material_data_ = new MaterialInfo*[material_count];
	object_data_ = new ObjectInfo*[object_count];

	memcpy_s(material_data_, material_count * sizeof(MaterialInfo*), material_data.data(), material_count * sizeof(MaterialInfo*));
	memcpy_s(object_data_, object_count * sizeof(ObjectInfo*), object_data.data(), object_count * sizeof(ObjectInfo*));

	world_ = new World(object_data_, material_data_, (int32_t)object_count, (int32_t)material_count);
}

void CpuRenderer::deallocate_world() const
{
	delete world_;
	delete[] object_data_;
	delete[] material_data_;
}