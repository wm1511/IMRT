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

Sphere::Sphere(const glm::dvec3 center = {0.0f, 0.0f, 0.0f}, const double radius = 0) : mCenter(center), mRadius(radius)
{
}

double Sphere::intersect(const Ray& ray) const
{
	const double b = dot(2.0 * (ray.origin - mCenter), ray.direction);
	const double c = dot(ray.origin - mCenter, ray.origin - mCenter) - mRadius * mRadius;
	double discriminant = b * b - 4 * c;
	if (discriminant < 0)
		return 0.0;
	discriminant = glm::sqrt(discriminant);
	const double solution1 = -b + discriminant;
	const double solution2 = -b - discriminant;
	return solution2 > std::numeric_limits<double>::epsilon()
		       ? solution2 / 2
		       : solution1 > std::numeric_limits<double>::epsilon()
		       ? solution1 / 2
		       : 0;
}

glm::dvec3 Sphere::normal(const glm::dvec3& point) const
{
    return (point - mCenter) / mRadius;
}

AABB Sphere::getAABB() const
{
	return {glm::dvec3(mCenter.x - mRadius, mCenter.y - mRadius, mCenter.z - mRadius),
			  glm::dvec3(mCenter.x + mRadius, mCenter.y + mRadius, mCenter.z + mRadius)};
}
