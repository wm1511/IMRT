#pragma once
#include "Object.hpp"

class Sphere final : public Object
{
public:
	Sphere(glm::dvec3 center, double radius);

	[[nodiscard]] double intersect(const Ray& ray) const override;
	[[nodiscard]] glm::dvec3 normal(const glm::dvec3& point) const override;
	[[nodiscard]] AABB getAABB() const override;

private:
	glm::dvec3 center_;
	double radius_;

};