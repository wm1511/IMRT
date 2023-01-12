#include "Sphere.hpp"

#include <limits>

Sphere::Sphere(const glm::dvec3 center = {0.0f, 0.0f, 0.0f}, const double radius = 0) : center_(center), radius_(radius)
{
}

double Sphere::intersect(const Ray& ray) const
{
	const glm::dvec3 oc = ray.origin_ - center_;
	const double b = dot(2.0 * oc, ray.direction_);
	const double c = dot(oc, oc) - radius_ * radius_;
	double discriminant = b * b - 4 * c;
	if (discriminant < 0)
		return 0.0;
	discriminant = glm::sqrt(discriminant);
	const double solution1 = -b + discriminant;
	const double solution2 = -b - discriminant;
	return solution2 > std::numeric_limits<double>::epsilon()
		       ? solution2 / 2
		       : solution1 > std::numeric_limits<double>::epsilon()
		       ? solution1 / 2
		       : 0;
}

glm::dvec3 Sphere::normal(const glm::dvec3& point) const
{
    return (point - center_) / radius_;
}

AABB Sphere::getAABB() const
{
	return {glm::dvec3(center_.x - radius_, center_.y - radius_, center_.z - radius_),
			  glm::dvec3(center_.x + radius_, center_.y + radius_, center_.z + radius_)};
}
