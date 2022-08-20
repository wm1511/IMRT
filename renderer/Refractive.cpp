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

double approximateSchlick(const double cosTh, const double ior)
{
	double r0 = (1.0 - ior) / (1.0 + ior);
	r0 = r0 * r0;
	return r0 + (1.0 - r0) * pow((1.0 - cosTh), 5);
}

glm::dvec3 Refractive::scatter(Ray& ray, const glm::dvec3 normal)
{
	const bool inside = dot(normal, ray.direction) > 0;
	const double ior = inside ? mIndexOfRefraction : 1.0 / mIndexOfRefraction;
	const glm::dvec3 n = inside ? -normal : normal;

	const double cosTheta1 = dot(n, ray.direction) * -1.0;
	const double cosTheta2 = 1.0 - ior * ior * (1.0 - cosTheta1 * cosTheta1);
	if (cosTheta2 > 0 && glm::linearRand(0.0, 1.0) > approximateSchlick(cosTheta1, ior))
		ray.direction = normalize(ray.direction * ior + n * (ior * cosTheta1 - glm::sqrt(cosTheta2)));
	else
		ray.direction = normalize(ray.direction + n * (cosTheta1 * 2));
	return glm::dvec3{1.0};
}
