#include "Renderer.hpp"

#include <glm/gtc/random.hpp>

#include <algorithm>

Renderer::Renderer(const RenderInfo* render_info) : render_info_(render_info)
{
	scene_.add(std::make_shared<Sphere>(glm::dvec3(0.0f, 0.0f, -1.0f), 0.5), std::make_shared<Diffuse>(glm::dvec3(1.0), 10000.0));
	scene_.add(std::make_shared<Sphere>(glm::dvec3(0.0f, -100.5f, -1.0f), 100.0), std::make_shared<Diffuse>(glm::dvec3(0.2f, 0.2f, 0.8f)));
}

void Renderer::render(float* image_data)
{
	const uint32_t width = render_info_->width;
    const uint32_t height = render_info_->height;
	scene_.RebuildBvh(1);

	const Camera camera({render_info_->look_origin[0], render_info_->look_origin[1], render_info_->look_origin[2]}, 
						{render_info_->look_target[0], render_info_->look_target[1], render_info_->look_target[2]}, 
						render_info_->fov,
	                    static_cast<double>(width) / static_cast<double>(height), 
						render_info_->aperture,
	                    render_info_->focus_distance);

 #pragma omp parallel for schedule(dynamic)
	for (int32_t y = 0; y < static_cast<int32_t>(height); y++)
	{
		for (int32_t x = 0; x < static_cast<int32_t>(width); x++)
		{
			glm::dvec3 pixel_color{0.0};
			for (int32_t k = 0; k < render_info_->samples_per_pixel; k++)
			{
				Ray ray = camera.cast_ray((x + glm::linearRand(0.0, 1.0)) / width, (y + glm::linearRand(0.0, 1.0)) / height);
				pixel_color += Trace(ray, scene_, 0) / static_cast<double>(render_info_->samples_per_pixel);
			}
			image_data[4 * (y * width + x)] = static_cast<float>(std::clamp(pixel_color.r, 0.0, 255.0)) / 255.0f;
			image_data[4 * (y * width + x) + 1] = static_cast<float>(std::clamp(pixel_color.g, 0.0, 255.0)) / 255.0f;
			image_data[4 * (y * width + x) + 2] = static_cast<float>(std::clamp(pixel_color.b, 0.0, 255.0)) / 255.0f;
			image_data[4 * (y * width + x) + 3] = 1.0f;
		}
	}
}

glm::dvec3 Renderer::Trace(Ray& ray, const Scene& scene, int32_t depth)
{
	glm::dvec3 color{};
	if (depth >= render_info_->max_depth)
		return glm::dvec3{0.0};

	const Intersection intersection = scene.intersect(ray);
	if (!intersection)
		return glm::dvec3{0.0};

	const glm::dvec3 hit_point = ray.origin_ + ray.direction_ * intersection.t_;
	const glm::dvec3 normal = intersection.object_->normal(hit_point);
	ray.origin_ = hit_point;

	const glm::dvec3 color_change = intersection.object_->material_->scatter(ray, normal);
	const auto material_emission = glm::dvec3{intersection.object_->material_->emission_};
	color += Trace(ray, scene, depth + 1) * color_change + material_emission;
	return color;
}
