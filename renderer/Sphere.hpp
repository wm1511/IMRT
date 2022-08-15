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

#pragma once
#include "Object.hpp"

class Sphere final : public Object
{
public:
	Sphere(glm::dvec3 center, double radius);

	[[nodiscard]] double intersect(const Ray& ray) const override;
	[[nodiscard]] glm::dvec3 normal(const glm::dvec3& point) const override;
	[[nodiscard]] AABB getAABB() const override;

private:
	glm::dvec3 mCenter;
	double mRadius;

};