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

Diffuse::Diffuse(const glm::vec3 color) : Material(color)
{
}

Diffuse::Diffuse(const glm::vec3 color, const float emission) : Material(color, emission)
{
}

void Diffuse::emit(Ray& ray, glm::vec3& colorChange, const glm::vec3 normal)
{
	glm::vec3 rotationX{0.0f}, rotationY{0.0f};
	Utils::orthonormalize(normal, rotationX, rotationY);
	const glm::vec3 randomDirection = Utils::hemisphereSample(glm::linearRand(0.0f, 1.0f), glm::linearRand(0.0f, 1.0f));
	glm::vec3 rotatedDirection;
	rotatedDirection.x = dot(glm::vec3(rotationX.x, rotationY.x, normal.x), randomDirection);
	rotatedDirection.y = dot(glm::vec3(rotationX.y, rotationY.y, normal.y), randomDirection);
	rotatedDirection.z = dot(glm::vec3(rotationX.z, rotationY.z, normal.z), randomDirection);
	ray.setDirection(rotatedDirection);
	const float cost = dot(ray.getDirection(), normal);
	colorChange = cost * this->getColor() * 0.1f;
}
