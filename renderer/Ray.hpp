#pragma once
#include <glm/glm.hpp>

class Ray
{
public:
	explicit Ray(const glm::dvec3 origin = glm::dvec3{0.0}, const glm::dvec3 direction = glm::dvec3{0.0}) :
		origin_(origin), direction_(normalize(direction))
	{
	}

	glm::dvec3 origin_, direction_;

};