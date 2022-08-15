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
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

class Ray
{
public:
	explicit Ray(const glm::dvec3 origin = glm::dvec3{0.0},
	             const glm::dvec3 direction = glm::dvec3{0.0}) : mOrigin(origin), mDirection(normalize(direction))
	{
	}

	[[nodiscard]] glm::dvec3 getOrigin() const { return mOrigin; }
	[[nodiscard]] glm::dvec3 getDirection() const { return mDirection; }

	void setOrigin(const glm::dvec3 origin) { mOrigin = origin; }
	void setDirection(const glm::dvec3 direction) { mDirection = direction; }

private:
	glm::dvec3 mOrigin, mDirection;

};