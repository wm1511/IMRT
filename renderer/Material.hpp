#pragma once
#include "Ray.hpp"

class Material
{
public:
	virtual ~Material() = default;

	explicit Material(const glm::dvec3 color) : color_(color)
	{
	}

	explicit Material(const glm::dvec3 color, const double emission) : color_(color), emission_(emission)
	{
	}

	virtual glm::dvec3 scatter(Ray& ray, glm::dvec3 normal) = 0;

	glm::dvec3 color_{0.0};
	double emission_ = 0.0;

};
