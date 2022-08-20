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

#include "Triangle.hpp"

#include <glm/ext/scalar_common.hpp>

double Triangle::intersect(const Ray& ray) const
{
	const glm::dvec3 v0v1 = mV1 - mV0;
	const glm::dvec3 v0v2 = mV2 - mV0;

	const glm::dvec3 pVec = cross(ray.direction, v0v2);

	const double determinant = dot(pVec, v0v1);

	if (abs(determinant) < std::numeric_limits<double>::epsilon())
		return 0.0;

	const double inverseDeterminant = 1.0 / determinant;

	const glm::dvec3 tVec = ray.origin - mV0;
	const double u = dot(pVec, tVec) * inverseDeterminant;
	if (u < 0 || u > 1)
		return 0.0;

	const glm::dvec3 qVec = cross(tVec, v0v1);
	const double v = dot(qVec, ray.direction) * inverseDeterminant;
	if (v < 0 || u + v > 1)
		return 0.0;

	return dot(qVec, v0v2) * inverseDeterminant;
}

glm::dvec3 Triangle::normal(const glm::dvec3&) const
{
	return cross(mV1 - mV0, mV2 - mV0);
}

AABB Triangle::getAABB() const
{
	return {{glm::min(mV0.x, mV1.x, mV2.x), glm::min(mV0.y, mV1.y, mV2.y), glm::min(mV0.z, mV1.z, mV2.z)},
		{glm::max(mV0.x, mV1.x, mV2.x), glm::max(mV0.y, mV1.y, mV2.y), glm::max(mV0.z, mV1.z, mV2.z)}};
}
