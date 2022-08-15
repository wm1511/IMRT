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

#include "Specular.hpp"

Specular::Specular(const glm::dvec3 color) : Material(color)
{
}

glm::dvec3 Specular::emit(Ray& ray, const glm::dvec3 normal)
{
	const double cost = dot(ray.getDirection(), normal);
	ray.setDirection(normalize(ray.getDirection() - normal * (cost * 2)));
	return glm::dvec3{1.0};
}
