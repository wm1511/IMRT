#include "Plane.hpp"

#include <limits>

Plane::Plane(const glm::dvec3 normal = {0.0, 0.0, 0.0}, const double offset = 0.0) : normal_(normal), offset_(offset)
{
}

double Plane::intersect(const Ray& ray) const
{
	const double angle = dot(normal_, ray.direction_);
	if (glm::abs(angle) > std::numeric_limits<double>::epsilon())
	{
		const double t = -1.0 * ((dot(normal_, ray.origin_) + offset_) / angle);
		return t > std::numeric_limits<double>::epsilon() ? t : 0;
	}
	return 0.0;
}

glm::dvec3 Plane::normal(const glm::dvec3&) const
{
	return normal_;
}

AABB Plane::getAABB() const
{
	constexpr double infinity = std::numeric_limits<double>::infinity();
	if (normal_.x == 0 && normal_.y == 0)
		return {glm::dvec3(-infinity, -infinity, offset_ * normal_.z), glm::dvec3(infinity, infinity, offset_ * normal_.z)};
	if (normal_.x == 0 && normal_.z == 0)
		return {glm::dvec3(-infinity, offset_ * normal_.y, -infinity), glm::dvec3(infinity, offset_ * normal_.y, infinity)};
	if (normal_.y == 0 && normal_.z == 0)
		return {glm::dvec3(offset_ * normal_.x, -infinity, -infinity), glm::dvec3(offset_ * normal_.z, infinity, infinity)};
	return {glm::dvec3{-infinity}, glm::dvec3{infinity}};
}
