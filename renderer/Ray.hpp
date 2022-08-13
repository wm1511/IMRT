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
	explicit Ray(const glm::vec3 origin = glm::vec3{0.0f}, const glm::vec3 direction = glm::vec3{0.0f}) : mOrigin(origin), mDirection(normalize(direction))
	{
	}

	[[nodiscard]] glm::vec3 getOrigin() const { return mOrigin; }
	[[nodiscard]] glm::vec3 getDirection() const { return mDirection; }

	void setOrigin(const glm::vec3 origin) { mOrigin = origin; }
	void setDirection(const glm::vec3 direction) { mDirection = direction; }

private:
	glm::vec3 mOrigin, mDirection;

};