#include "Diffuse.hpp"

#include <glm/gtc/random.hpp>

Diffuse::Diffuse(const glm::dvec3 color) : Material(color)
{
}

Diffuse::Diffuse(const glm::dvec3 color, const double emission) : Material(color, emission)
{
}

glm::dvec3 Diffuse::scatter(Ray& ray, const glm::dvec3 normal)
{
	ray.direction_ = normalize(glm::sphericalRand(1.0) + normal);
	const double cost = dot(ray.direction_, normal);
	return cost * this->color_;
}
