#pragma once
#include "Material.hpp"
#include "AABB.hpp"

#include <memory>

class Object
{
public:
	virtual ~Object() = default;

	[[nodiscard]] virtual double intersect(const Ray&) const = 0;
	[[nodiscard]] virtual glm::dvec3 normal(const glm::dvec3&) const = 0;
	[[nodiscard]] virtual AABB getAABB() const = 0;

	std::shared_ptr<Material> material_;

};