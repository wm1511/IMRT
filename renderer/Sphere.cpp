// Copyright (c) 2022, Wiktor Merta
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "Sphere.hpp"

#include <limits>

Sphere::Sphere(const glm::vec3 center = {0.0f, 0.0f, 0.0f}, const float radius = 0) : mCenter(center), mRadius(radius)
{
}

float Sphere::intersect(const Ray& ray) const
{
	const float b = dot(2.0f * (ray.getOrigin() - mCenter), ray.getDirection());
	const float c = dot(ray.getOrigin() - mCenter, ray.getOrigin() - mCenter) - mRadius * mRadius;
	float discriminant = b * b - 4 * c;
	if (discriminant < 0)
		return 0.0f;
	discriminant = glm::sqrt(discriminant);
	const float solution1 = -b + discriminant;
	const float solution2 = -b - discriminant;
	return solution2 > std::numeric_limits<float>::epsilon()
		       ? solution2 / 2
		       : solution1 > std::numeric_limits<float>::epsilon()
		       ? solution1 / 2
		       : 0;
}

glm::vec3 Sphere::normal(const glm::vec3& point) const
{
    return normalize(point - mCenter);
}

AABB Sphere::getAABB() const
{
	return {glm::vec3(mCenter.x - mRadius, mCenter.y - mRadius, mCenter.z - mRadius),
			  glm::vec3(mCenter.x + mRadius, mCenter.y + mRadius, mCenter.z + mRadius)};
}
