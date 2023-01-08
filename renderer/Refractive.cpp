#include "Refractive.hpp"

#include <glm/gtc/random.hpp>

Refractive::Refractive(const glm::dvec3 color, const double index_of_refraction) : Material(color), index_of_refraction_(index_of_refraction)
{
}

double approximateSchlick(const double cos_theta, const double ior)
{
	double r0 = (1.0 - ior) / (1.0 + ior);
	r0 = r0 * r0;
	return r0 + (1.0 - r0) * pow((1.0 - cos_theta), 5);
}

glm::dvec3 Refractive::scatter(Ray& ray, const glm::dvec3 normal)
{
	const bool inside = dot(normal, ray.direction_) > 0;
	const double ior = inside ? index_of_refraction_ : 1.0 / index_of_refraction_;
	const glm::dvec3 n = inside ? -normal : normal;

	const double cos_theta1 = dot(n, ray.direction_) * -1.0;
	const double cos_theta2 = 1.0 - ior * ior * (1.0 - cos_theta1 * cos_theta1);
	if (cos_theta2 > 0 && glm::linearRand(0.0, 1.0) > approximateSchlick(cos_theta1, ior))
		ray.direction_ = normalize(ray.direction_ * ior + n * (ior * cos_theta1 - glm::sqrt(cos_theta2)));
	else
		ray.direction_ = normalize(ray.direction_ + n * (cos_theta1 * 2));
	return glm::dvec3{1.0};
}
