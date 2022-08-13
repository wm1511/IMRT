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

#include <limits>

class AABB
{
public:
	AABB() : mMin(glm::vec3{std::numeric_limits<float>::infinity()}), mMax(glm::vec3{-std::numeric_limits<float>::infinity()})
	{
	}

	AABB(const glm::vec3 min, const glm::vec3 max) : mMin(min), mMax(max)
	{	
	}


	[[nodiscard]] bool unbounded() const
	{
		return mMin.x == -std::numeric_limits<float>::infinity() ||
			mMin.y == -std::numeric_limits<float>::infinity() ||
			mMin.z == -std::numeric_limits<float>::infinity() ||
			mMax.x == std::numeric_limits<float>::infinity() ||
			mMax.y == std::numeric_limits<float>::infinity() ||
			mMax.z == std::numeric_limits<float>::infinity();
	}

	[[nodiscard]] uint64_t getLargestDimension() const
	{
		const float dx = glm::abs(mMax.x - mMin.x);
		const float dy = glm::abs(mMax.y - mMin.y);
		const float dz = glm::abs(mMax.z - mMin.z);
		
		if (dx > dy && dx > dz)
			return 0;
		if (dy > dz)
			return 1;
		return 2;
	}

	[[nodiscard]] bool intersect(const Ray& ray, const glm::vec3& inverseDirection, float closestT) const
	{
		float txmin = ((ray.getDirection().x < 0 ? mMax.x : mMin.x) - ray.getOrigin().x) * inverseDirection.x;
		float txmax = ((ray.getDirection().x < 0 ? mMin.x : mMax.x) - ray.getOrigin().x) * inverseDirection.x;
		const float tymin = ((ray.getDirection().y < 0 ? mMax.y : mMin.y) - ray.getOrigin().y) * inverseDirection.y;
		const float tymax = ((ray.getDirection().y < 0 ? mMin.y : mMax.y) - ray.getOrigin().y) * inverseDirection.y;

		if (txmin > tymax || tymin > txmax)
			return false;
		if (tymin > txmin)
			txmin = tymin;
		if (tymax < txmax)
			txmax = tymax;

		const float tzmin = ((ray.getDirection().z < 0 ? mMax.z : mMin.z) - ray.getOrigin().z) * inverseDirection.z;
		const float tzmax = ((ray.getDirection().z < 0 ? mMin.z : mMax.z) - ray.getOrigin().z) * inverseDirection.z;

		if (txmin > tzmax || tzmin > txmax)
			return false;
		if (tzmin > txmin)
			txmin = tzmin;
		if (tzmax < txmax)
			txmax = tzmax;
		return txmin < closestT && txmax > std::numeric_limits<float>::epsilon();
	}

	[[nodiscard]] glm::vec3 getMin() const { return mMin; }
	[[nodiscard]] glm::vec3 getMax() const { return mMax; }

	void enclose(const AABB& other)
	{
		this->mMin.x = glm::min(this->mMin.x, other.mMin.x);
		this->mMin.y = glm::min(this->mMin.y, other.mMin.y);
		this->mMin.z = glm::min(this->mMin.z, other.mMin.z);

		this->mMax.x = glm::max(this->mMax.x, other.mMax.x);
		this->mMax.y = glm::max(this->mMax.y, other.mMax.y);
		this->mMax.z = glm::max(this->mMax.z, other.mMax.z);
	}

	void enclose(const AABB& first, const AABB& second)
	{
		this->mMin.x = glm::min(first.mMin.x, second.mMin.x);
		this->mMin.y = glm::min(first.mMin.y, second.mMin.y);
		this->mMin.z = glm::min(first.mMin.z, second.mMin.z);

		this->mMax.x = glm::max(first.mMax.x, second.mMax.x);
		this->mMax.y = glm::max(first.mMax.y, second.mMax.y);
		this->mMax.z = glm::max(first.mMax.z, second.mMax.z);
	}

	void enclose(const glm::vec3& point)
	{
		this->mMin.x = glm::min(this->mMin.x, point.x);
		this->mMin.y = glm::min(this->mMin.y, point.y);
		this->mMin.z = glm::min(this->mMin.z, point.z);

		this->mMax.x = glm::max(this->mMax.x, point.x);
		this->mMax.y = glm::max(this->mMax.y, point.y);
		this->mMax.z = glm::max(this->mMax.z, point.z);
	}

private:
	glm::vec3 mMin;
	glm::vec3 mMax;
};