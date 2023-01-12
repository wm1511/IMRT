#pragma once
#include "Ray.hpp"

#include "glm/gtc/random.hpp"

class Camera
{
public:
	Camera(glm::dvec3 look_origin, glm::dvec3 look_target, float fov, double aspect_ratio, double aperture, double focus_distance)
	{
		double viewport_height = 2.0 * static_cast<double>(glm::tan(fov / 2));
		double viewport_width = viewport_height * aspect_ratio;

		glm::dvec3 camera_direction = normalize(look_origin - look_target);
		u_ = normalize(cross({0.0, -1.0, 0.0}, camera_direction));
		v_ = cross(camera_direction, u_);

		origin_ = look_origin;
		horizontal_map_ = focus_distance * viewport_width * u_;
		vertical_map_ = focus_distance * viewport_height * v_;
		start_ = origin_ - horizontal_map_ / 2.0 - vertical_map_ / 2.0 - focus_distance * camera_direction;
		lens_radius_ = aperture / 2.0;
	}

	[[nodiscard]] Ray cast_ray(const double x, const double y) const
	{
		const glm::dvec3 random_on_lens = lens_radius_ * normalize(
			glm::dvec3(glm::linearRand(-1.0, 1.0), glm::linearRand(-1.0, 1.0), 0.0));
		const glm::dvec3 offset = u_ * random_on_lens.x + v_ * random_on_lens.y;
		return Ray(origin_ + offset, start_ + x * horizontal_map_ + y * vertical_map_ - origin_ - offset);
	}

private:
	glm::dvec3 origin_{};
	glm::dvec3 start_{};
	glm::dvec3 horizontal_map_{};
	glm::dvec3 vertical_map_{};
	glm::dvec3 u_{}, v_{};
	double lens_radius_{};

};