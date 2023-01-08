#include "Triangle.hpp"

#include <glm/ext/scalar_common.hpp>

double Triangle::intersect(const Ray& ray) const
{
	const glm::dvec3 v0_v1 = v1_ - v0_;
	const glm::dvec3 v0_v2 = v2_ - v0_;

	const glm::dvec3 p_vec = cross(ray.direction_, v0_v2);

	const double determinant = dot(p_vec, v0_v1);

	if (abs(determinant) < std::numeric_limits<double>::epsilon())
		return 0.0;

	const double inverse_determinant = 1.0 / determinant;

	const glm::dvec3 t_vec = ray.origin_ - v0_;
	const double u = dot(p_vec, t_vec) * inverse_determinant;
	if (u < 0 || u > 1)
		return 0.0;

	const glm::dvec3 q_vec = cross(t_vec, v0_v1);
	const double v = dot(q_vec, ray.direction_) * inverse_determinant;
	if (v < 0 || u + v > 1)
		return 0.0;

	return dot(q_vec, v0_v2) * inverse_determinant;
}

glm::dvec3 Triangle::normal(const glm::dvec3&) const
{
	return cross(v1_ - v0_, v2_ - v0_);
}

AABB Triangle::getAABB() const
{
	return {{glm::min(v0_.x, v1_.x, v2_.x), glm::min(v0_.y, v1_.y, v2_.y), glm::min(v0_.z, v1_.z, v2_.z)},
		{glm::max(v0_.x, v1_.x, v2_.x), glm::max(v0_.y, v1_.y, v2_.y), glm::max(v0_.z, v1_.z, v2_.z)}};
}
