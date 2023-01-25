#pragma once
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

class Ray
{
	public:
    Ray(const glm::vec3& origin, const glm::vec3& direction) : origin_(origin), direction_(direction) {}
    glm::vec3 origin() const { return origin_; }
    glm::vec3 direction() const { return direction_; }
    glm::vec3 position(const float t) const { return origin_ + t * direction_; }

private:
    glm::vec3 origin_;
    glm::vec3 direction_;
};