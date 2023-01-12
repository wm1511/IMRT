#include "Renderer.hpp"

#include "Camera.hpp"

#include <glm/gtc/random.hpp>

#include <thread>
#include <algorithm>
#include <functional>

Renderer::Renderer(const RtInfo* rt_info) : rt_info_(rt_info)
{
}

void Renderer::render(float* image_data, const uint32_t width, const uint32_t height)
{
	if (rt_info_->scene_index == 0)
		prepare_scene_ = &SceneBuilder::MakeCornellBox;
	else if (rt_info_->scene_index == 1)
		prepare_scene_ = &SceneBuilder::MakeUkraine;
	else if (rt_info_->scene_index == 2)
		prepare_scene_ = &SceneBuilder::MakeChoinka;

	if (rt_info_->trace_type == 0)
		trace_ = &Renderer::DTrace;
	else if (rt_info_->trace_type == 1)
		trace_ = &Renderer::RrTrace;

	Scene scene = prepare_scene_();
	scene.RebuildBvh(1);

	const Camera camera({rt_info_->look_origin_x, rt_info_->look_origin_y, rt_info_->look_origin_z}, 
						{rt_info_->look_target_x, rt_info_->look_target_y, rt_info_->look_target_z}, 
						rt_info_->fov,
	                    static_cast<double>(width) / static_cast<double>(height), 
						rt_info_->aperture,
	                    rt_info_->focus_distance);

 #pragma omp parallel for schedule(dynamic)
	for (int32_t y = 0; y < static_cast<int32_t>(height); y++)
	{
		for (int32_t x = 0; x < static_cast<int32_t>(width); x++)
		{
			glm::dvec3 pixel_color{0.0};
			for (int32_t k = 0; k < rt_info_->samples_per_pixel; k++)
			{
				Ray ray = camera.cast_ray((x + glm::linearRand(0.0, 1.0)) / width, (y + glm::linearRand(0.0, 1.0)) / height);
				pixel_color += trace_(this, ray, scene, 0) / static_cast<double>(rt_info_->samples_per_pixel);
			}
			image_data[4 * (y * width + x)] = static_cast<float>(std::clamp(pixel_color.r, 0.0, 255.0)) / 255.0f;
			image_data[4 * (y * width + x) + 1] = static_cast<float>(std::clamp(pixel_color.g, 0.0, 255.0)) / 255.0f;
			image_data[4 * (y * width + x) + 2] = static_cast<float>(std::clamp(pixel_color.b, 0.0, 255.0)) / 255.0f;
			image_data[4 * (y * width + x) + 3] = 1.0f;
		}
	}
}

glm::dvec3 Renderer::RrTrace(Ray& ray, const Scene& scene, int32_t depth)
{
	glm::dvec3 color{};
	double rr_factor = 1.0;
	if (depth >= rt_info_->rr_certain_depth)
	{
		if (glm::linearRand(0.0, 1.0) <= rt_info_->rr_stop_probability)
			return glm::dvec3{0.0};
		rr_factor = 1.0 / (1.0 - rt_info_->rr_stop_probability);
	}

	const Intersection intersection = scene.intersect(ray);
	if (!intersection)
		return glm::dvec3{0.0};

	const glm::dvec3 hit_point = ray.origin_ + ray.direction_ * intersection.t_;
	const glm::dvec3 normal = intersection.object_->normal(hit_point);
	ray.origin_ = hit_point;

	const glm::dvec3 color_change = intersection.object_->material_->scatter(ray, normal);
	const auto material_emission = glm::dvec3{intersection.object_->material_->emission_};
	color += (RrTrace(ray, scene, depth + 1) * color_change + material_emission) * rr_factor;
	return color;
}

glm::dvec3 Renderer::DTrace(Ray& ray, const Scene& scene, int32_t depth)
{
	glm::dvec3 color{};
	if (depth >= rt_info_->max_depth)
		return glm::dvec3{0.0};

	const Intersection intersection = scene.intersect(ray);
	if (!intersection)
		return glm::dvec3{0.0};

	const glm::dvec3 hit_point = ray.origin_ + ray.direction_ * intersection.t_;
	const glm::dvec3 normal = intersection.object_->normal(hit_point);
	ray.origin_ = hit_point;

	const glm::dvec3 color_change = intersection.object_->material_->scatter(ray, normal);
	const auto material_emission = glm::dvec3{intersection.object_->material_->emission_};
	color += DTrace(ray, scene, depth + 1) * color_change + material_emission;
	return color;
}
