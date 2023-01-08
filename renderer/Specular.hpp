#pragma once
#include "Material.hpp"

class Specular final : public Material
{
public:
	explicit Specular(glm::dvec3 color, double blurriness);

	glm::dvec3 scatter(Ray& ray, glm::dvec3 normal) override;

private:
	double blurriness_ = 0.0;

};