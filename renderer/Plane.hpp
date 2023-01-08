#pragma once
#include "Object.hpp"

class Plane final : public Object
{
public:
	Plane(glm::dvec3 normal, double offset);

	[[nodiscard]] double intersect(const Ray& ray) const override;
	[[nodiscard]] glm::dvec3 normal(const glm::dvec3&) const override;
	[[nodiscard]] AABB getAABB() const override;

private:
	glm::dvec3 normal_;
	double offset_;
};