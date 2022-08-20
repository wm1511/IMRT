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

#include <glm/gtc/random.hpp>

Diffuse::Diffuse(const glm::dvec3 color) : Material(color)
{
}

Diffuse::Diffuse(const glm::dvec3 color, const double emission) : Material(color, emission)
{
}

glm::dvec3 Diffuse::scatter(Ray& ray, const glm::dvec3 normal)
{
	ray.direction = normalize(glm::sphericalRand(1.0) + normal);
	const double cost = dot(ray.direction, normal);
	return cost * this->color;
}
