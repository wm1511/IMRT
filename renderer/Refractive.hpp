#pragma once
#include "Material.hpp"

class Refractive final : public Material
{
public:
	Refractive(glm::dvec3 color, double index_of_refraction);

	glm::dvec3 scatter(Ray& ray, glm::dvec3 normal) override;

private:
	double index_of_refraction_ = 1.0;

};