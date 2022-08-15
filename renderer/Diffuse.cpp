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

#include "Diffuse.hpp"
#include "Utils.hpp"

#include <glm/gtc/random.hpp>

Diffuse::Diffuse(const glm::dvec3 color) : Material(color)
{
}

Diffuse::Diffuse(const glm::dvec3 color, const double emission) : Material(color, emission)
{
}

glm::dvec3 Diffuse::emit(Ray& ray, const glm::dvec3 normal)
{
	glm::dvec3 rotationX{0.0}, rotationY{0.0};
	Utils::orthonormalize(normal, rotationX, rotationY);
	const glm::dvec3 randomDirection = Utils::hemisphereSample(glm::linearRand(0.0, 1.0), glm::linearRand(0.0, 1.0));
	glm::dvec3 rotatedDirection;
	rotatedDirection.x = dot(glm::dvec3(rotationX.x, rotationY.x, normal.x), randomDirection);
	rotatedDirection.y = dot(glm::dvec3(rotationX.y, rotationY.y, normal.y), randomDirection);
	rotatedDirection.z = dot(glm::dvec3(rotationX.z, rotationY.z, normal.z), randomDirection);
	ray.setDirection(rotatedDirection);
	const double cost = dot(ray.getDirection(), normal);
	return cost * this->getColor() * 0.1;
}
