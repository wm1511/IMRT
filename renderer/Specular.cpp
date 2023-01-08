#include "Specular.hpp"

#include <glm/gtc/random.hpp>

Specular::Specular(const glm::dvec3 color, const double blurriness) : Material(color), blurriness_(blurriness)
{
}

glm::dvec3 Specular::scatter(Ray& ray, const glm::dvec3 normal)
{
	const double cost = dot(ray.direction_, normal);
	ray.direction_ = normalize(ray.direction_ - normal * 2.0 * cost + blurriness_ * glm::sphericalRand(1.0));
	return glm::dvec3{1.0};
}
