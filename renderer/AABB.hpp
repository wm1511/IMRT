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
	AABB() : mMin(glm::dvec3{std::numeric_limits<double>::infinity()}), mMax(glm::dvec3{-std::numeric_limits<double>::infinity()})
	{
	}

	AABB(const glm::dvec3 min, const glm::dvec3 max) : mMin(min), mMax(max)
	{	
	}


	[[nodiscard]] bool unbounded() const
	{
		return mMin.x == -std::numeric_limits<double>::infinity() ||
			mMin.y == -std::numeric_limits<double>::infinity() ||
			mMin.z == -std::numeric_limits<double>::infinity() ||
			mMax.x == std::numeric_limits<double>::infinity() ||
			mMax.y == std::numeric_limits<double>::infinity() ||
			mMax.z == std::numeric_limits<double>::infinity();
	}

	[[nodiscard]] uint64_t getLargestDimension() const
	{
		const double dx = glm::abs(mMax.x - mMin.x);
		const double dy = glm::abs(mMax.y - mMin.y);
		const double dz = glm::abs(mMax.z - mMin.z);
		
		if (dx > dy && dx > dz)
			return 0;
		if (dy > dz)
			return 1;
		return 2;
	}

	[[nodiscard]] bool intersect(const Ray& ray, const glm::dvec3& inverseDirection, const double closestT) const
	{
		double txmax = ((ray.direction.x < 0 ? mMin.x : mMax.x) - ray.origin.x) * inverseDirection.x;
		double txmin = ((ray.direction.x < 0 ? mMax.x : mMin.x) - ray.origin.x) * inverseDirection.x;
		const double tymin = ((ray.direction.y < 0 ? mMax.y : mMin.y) - ray.origin.y) * inverseDirection.y;
		const double tymax = ((ray.direction.y < 0 ? mMin.y : mMax.y) - ray.origin.y) * inverseDirection.y;

		if (txmin > tymax || tymin > txmax)
			return false;
		if (tymin > txmin)
			txmin = tymin;
		if (tymax < txmax)
			txmax = tymax;

		const double tzmin = ((ray.direction.z < 0 ? mMax.z : mMin.z) - ray.origin.z) * inverseDirection.z;
		const double tzmax = ((ray.direction.z < 0 ? mMin.z : mMax.z) - ray.origin.z) * inverseDirection.z;

		if (txmin > tzmax || tzmin > txmax)
			return false;
		if (tzmin > txmin)
			txmin = tzmin;
		if (tzmax < txmax)
			txmax = tzmax;
		return txmin < closestT && txmax > std::numeric_limits<double>::epsilon();
	}

	[[nodiscard]] glm::dvec3 getMin() const { return mMin; }
	[[nodiscard]] glm::dvec3 getMax() const { return mMax; }

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

	void enclose(const glm::dvec3& point)
	{
		this->mMin.x = glm::min(this->mMin.x, point.x);
		this->mMin.y = glm::min(this->mMin.y, point.y);
		this->mMin.z = glm::min(this->mMin.z, point.z);

		this->mMax.x = glm::max(this->mMax.x, point.x);
		this->mMax.y = glm::max(this->mMax.y, point.y);
		this->mMax.z = glm::max(this->mMax.z, point.z);
	}

private:
	glm::dvec3 mMin;
	glm::dvec3 mMax;
};