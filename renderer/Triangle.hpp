#pragma once
#include "Object.hpp"

class Triangle final : public Object
{
public:
	Triangle(const glm::dvec3 v0, const glm::dvec3 v1, const glm::dvec3 v2) : v0_(v0), v1_(v1), v2_(v2)
	{
	}

	[[nodiscard]] double intersect(const Ray& ray) const override;
	[[nodiscard]] glm::dvec3 normal(const glm::dvec3&) const override;
	[[nodiscard]] AABB getAABB() const override;

private:
	glm::dvec3 v0_, v1_, v2_;
};
