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
#include "Ray.hpp"

class Material
{
public:
	virtual ~Material() = default;

	explicit Material(const glm::dvec3 color) : color(color)
	{
	}

	explicit Material(const glm::dvec3 color, const double emission) : color(color), emission(emission)
	{
	}

	virtual glm::dvec3 scatter(Ray& ray, glm::dvec3 normal) = 0;

	glm::dvec3 color{0.0};
	double emission = 0.0;

};
