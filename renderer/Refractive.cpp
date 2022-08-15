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

#include "Refractive.hpp"

#include <glm/gtc/random.hpp>

Refractive::Refractive(const glm::dvec3 color, const double indexOfRefraction) : Material(color), mIndexOfRefraction(indexOfRefraction)
{
}

glm::dvec3 Refractive::emit(Ray& ray, const glm::dvec3 normal)
{
	double ior = this->getIndexOfRefraction();
	glm::dvec3 n = normal;
	double r0 = (1.0 - ior) / (1.0 + ior);
	r0 = r0 * r0;
	if (dot(normal, ray.getDirection()) > 0)
	{
		n = normal * -1.0;
		ior = 1 / ior;
	}
	ior = 1 / ior;
	const double cosTheta1 = dot(n, ray.getDirection()) * -1.0;
	const double cosTheta2 = 1.0 - ior * ior * (1.0 - cosTheta1 * cosTheta1);
	const double reflCoeff = r0 + (1.0 - r0) * glm::pow(1.0 - cosTheta1, 5.0);
	if (cosTheta2 > 0 && glm::linearRand(0.0, 1.0) > reflCoeff)
		ray.setDirection(normalize(ray.getDirection() * ior + n * (ior * cosTheta1 - glm::sqrt(cosTheta2))));
	else
		ray.setDirection(normalize(ray.getDirection() + n * (cosTheta1 * 2)));
	return glm::dvec3{1.15};
}
