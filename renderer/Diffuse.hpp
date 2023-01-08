#pragma once
#include "Material.hpp"

class Diffuse final : public Material
{
public:
	explicit Diffuse(glm::dvec3 color);
	explicit Diffuse(glm::dvec3 color, double emission);

	glm::dvec3 scatter(Ray& ray, glm::dvec3 normal) override;

};