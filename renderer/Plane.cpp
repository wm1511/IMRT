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

#include "Plane.hpp"

#include <limits>

Plane::Plane(const glm::dvec3 normal = {0.0, 0.0, 0.0}, const double offset = 0.0) : mNormal(normal), mOffset(offset)
{
}

double Plane::intersect(const Ray& ray) const
{
	const double angle = dot(mNormal, ray.direction);
	if (glm::abs(angle) > std::numeric_limits<double>::epsilon())
	{
		const double t = -1.0 * ((dot(mNormal, ray.origin) + mOffset) / angle);
		return t > std::numeric_limits<double>::epsilon() ? t : 0;
	}
	return 0.0;
}

glm::dvec3 Plane::normal(const glm::dvec3&) const
{
	return mNormal;
}

AABB Plane::getAABB() const
{
	constexpr double infinity = std::numeric_limits<double>::infinity();
	if (mNormal.x == 0 && mNormal.y == 0)
		return {glm::dvec3(-infinity, -infinity, mOffset * mNormal.z), glm::dvec3(infinity, infinity, mOffset * mNormal.z)};
	if (mNormal.x == 0 && mNormal.z == 0)
		return {glm::dvec3(-infinity, mOffset * mNormal.y, -infinity), glm::dvec3(infinity, mOffset * mNormal.y, infinity)};
	if (mNormal.y == 0 && mNormal.z == 0)
		return {glm::dvec3(mOffset * mNormal.x, -infinity, -infinity), glm::dvec3(mOffset * mNormal.z, infinity, infinity)};
	return {glm::dvec3{-infinity}, glm::dvec3{infinity}};
}
