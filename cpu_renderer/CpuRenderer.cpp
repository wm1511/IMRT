#include "CpuRenderer.hpp"

#include <cstring>

CpuRenderer::CpuRenderer(const RenderInfo* render_info, const WorldInfo* world_info) : render_info_(render_info), world_info_(world_info)
{
	const uint64_t image_size = (uint64_t)render_info_->width * render_info_->height;

	accumulation_buffer_ = new float4[image_size];
	xoshiro_state_ = new uint4[image_size];

	random_init();
	allocate_world();

	camera_ = new Camera*;
	*camera_ = new Camera(
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

	delete *camera_;
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
				const Ray ray = (*camera_)->cast_ray(&local_random_state, u, v);
				const float3 color = sqrt(calculate_color(ray, world_, hdr_data_, *render_info_, &local_random_state));

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
					const Ray ray = (*camera_)->cast_ray(&local_random_state, u, v);
					const float3 color = sqrt(calculate_color(ray, world_, hdr_data_, *render_info_, &local_random_state));

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
	(*camera_)->update(
		render_info_->camera_position,
		render_info_->camera_direction,
		render_info_->fov,
		render_info_->aperture,
		render_info_->focus_distance);
}

void CpuRenderer::refresh_world()
{
	for (int32_t i = 0; i < world_info_->material_count; i++)
	{
	   if (world_info_->material_data[i]->type == DIFFUSE)
		   ((Diffuse*)materials_list_[i])->update((DiffuseInfo*)world_info_->material_data[i]);
	   else if (world_info_->material_data[i]->type == SPECULAR)
		   ((Specular*)materials_list_[i])->update((SpecularInfo*)world_info_->material_data[i]);
	   else if (world_info_->material_data[i]->type == REFRACTIVE)
		   ((Refractive*)materials_list_[i])->update((RefractiveInfo*)world_info_->material_data[i]);
	}
	
	for (int32_t i = 0; i < world_info_->object_count; i++)
	{
	   if (world_info_->object_data[i]->type == SPHERE)
		   ((Sphere*)primitives_list_[i])->update((SphereInfo*)world_info_->object_data[i], materials_list_[world_info_->object_data[i]->material_id]);
	   else if (world_info_->object_data[i]->type == TRIANGLE)
		   ((Triangle*)primitives_list_[i])->update((TriangleInfo*)world_info_->object_data[i], materials_list_[world_info_->object_data[i]->material_id]);
	}

	(*world_)->update(world_info_->object_count);
}

void CpuRenderer::recreate_camera()
{
	delete *camera_;
	*camera_ = new Camera(
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
	primitives_list_ = new Primitive*[world_info_->object_count];
	materials_list_ = new Material*[world_info_->material_count];
	world_ = new World*;

	MaterialInfo** material_data = world_info_->material_data;
    ObjectInfo** object_data = world_info_->object_data;

	for (int32_t i = 0; i < world_info_->material_count; i++)
	{
		if (material_data[i]->type == DIFFUSE)
			materials_list_[i] = new Diffuse((DiffuseInfo*)material_data[i]);
		else if (material_data[i]->type == SPECULAR)
			materials_list_[i] = new Specular((SpecularInfo*)material_data[i]);
		else if (material_data[i]->type == REFRACTIVE)
			materials_list_[i] = new Refractive((RefractiveInfo*)material_data[i]);
	}

	 for (int32_t i = 0; i < world_info_->object_count; i++)
	 {
		if (object_data[i]->type == SPHERE)
			primitives_list_[i] = new Sphere((SphereInfo*)object_data[i], materials_list_[object_data[i]->material_id]);
		else if (object_data[i]->type == TRIANGLE)
			primitives_list_[i] = new Triangle((TriangleInfo*)object_data[i], materials_list_[object_data[i]->material_id]);
	 }

	*world_ = new World(primitives_list_, world_info_->object_count);
}

void CpuRenderer::deallocate_world() const
{
	for (int32_t i = 0; i < world_info_->object_count; i++)
		delete materials_list_[i];

	for (int32_t i = 0; i < world_info_->object_count; i++)
		delete primitives_list_[i];

	delete world_;
	delete[] primitives_list_;
	delete[] materials_list_;
}