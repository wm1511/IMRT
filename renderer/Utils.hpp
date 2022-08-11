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
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

class Utils
{
public:
	Utils() = delete;

	[[nodiscard]] static glm::vec3 hemisphereSample(const float r1, const float r2)
	{
		const float radius = glm::sqrt(1.0f - r1 * r1);
		const float angle = 2 * glm::pi<float>() * r2;
		return {glm::cos(angle) * radius, glm::sin(angle) * radius, r1};
	}

	[[nodiscard]] static glm::vec3 getCameraCoords(const uint32_t x, const uint32_t y, const uint32_t width, const uint32_t height)
	{
		const auto w = static_cast<float>(width);
		const auto h = static_cast<float>(height);
		constexpr float fovx = glm::pi<float>() / 4;
		const float fovy = h / w * fovx;
		return {(static_cast<float>(2 * x) - w) / w * glm::tan(fovx), -((static_cast<float>(2 * y) - h) / h) * glm::tan(fovy), -1.0f};
	}

	static uint32_t convert(const glm::vec3 color)
	{
		return 0xff000000 |
			static_cast<uint8_t>(color.z) << 16 |
			static_cast<uint8_t>(color.y) << 8 |
			static_cast<uint8_t>(color.x);
	}

	static void orthonormalize(const glm::vec3& v1, glm::vec3& v2, glm::vec3& v3)
	{
		if (glm::abs(v1.x) > glm::abs(v1.y))
		{
			const float inverseLength = 1.0f / glm::sqrt(v1.x * v1.x + v1.z * v1.z);
			v2 = glm::vec3(-v1.z * inverseLength, 0.0f, v1.x * inverseLength);
		}
		else
		{
			const float inverseLength = 1.0f / glm::sqrt(v1.y * v1.y + v1.z * v1.z);
			v2 = glm::vec3(0.0f, v1.z * inverseLength, -v1.y * inverseLength);
		}
		v3 = cross(v1, v2);
	}
};

