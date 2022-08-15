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

	explicit Material(const glm::dvec3 color) : mColor(color)
	{
	}

	explicit Material(const glm::dvec3 color, const double emission) : mColor(color), mEmission(emission)
	{
	}

	virtual glm::dvec3 emit(Ray& ray, glm::dvec3 normal) = 0;

	[[nodiscard]] double getEmission() const { return mEmission; }
	[[nodiscard]] glm::dvec3 getColor() const { return mColor; }

protected:
	glm::dvec3 mColor{0.0};
	double mEmission = 0.0;

};
