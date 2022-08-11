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

#include "Square.hpp"

#include <limits>

Square::Square(const glm::vec3 normal = {0.0f, 0.0f, 0.0f}, const float offset = 0.0f) : mNormal(normal), mOffset(offset)
{
}

float Square::intersect(const Ray& ray) const
{
	const float angle = dot(mNormal, ray.getDirection());
	if (glm::abs(angle) > std::numeric_limits<float>::epsilon())
	{
		const float t = -1.0f * ((dot(mNormal, ray.getOrigin()) + mOffset) / angle);
		return t > std::numeric_limits<float>::epsilon() ? t : 0;
	}
	return 0.0f;
}

glm::vec3 Square::normal(const glm::vec3&) const
{
	return mNormal;
}
