#pragma once
#include "Ray.hpp"

#include <limits>

class AABB
{
public:
	AABB() : min_(glm::dvec3{std::numeric_limits<double>::infinity()}), max_(glm::dvec3{-std::numeric_limits<double>::infinity()})
	{
	}

	AABB(const glm::dvec3 min, const glm::dvec3 max) : min_(min), max_(max)
	{	
	}


	[[nodiscard]] bool unbounded() const
	{
		return min_.x == -std::numeric_limits<double>::infinity() ||
			min_.y == -std::numeric_limits<double>::infinity() ||
			min_.z == -std::numeric_limits<double>::infinity() ||
			max_.x == std::numeric_limits<double>::infinity() ||
			max_.y == std::numeric_limits<double>::infinity() ||
			max_.z == std::numeric_limits<double>::infinity();
	}

	[[nodiscard]] uint64_t GetLargestDimension() const
	{
		const double dx = glm::abs(max_.x - min_.x);
		const double dy = glm::abs(max_.y - min_.y);
		const double dz = glm::abs(max_.z - min_.z);
		
		if (dx > dy && dx > dz)
			return 0;
		if (dy > dz)
			return 1;
		return 2;
	}

	[[nodiscard]] bool intersect(const Ray& ray, const glm::dvec3& inverse_direction, const double closest_t) const
	{
		double txmax = ((ray.direction_.x < 0 ? min_.x : max_.x) - ray.origin_.x) * inverse_direction.x;
		double txmin = ((ray.direction_.x < 0 ? max_.x : min_.x) - ray.origin_.x) * inverse_direction.x;
		const double tymin = ((ray.direction_.y < 0 ? max_.y : min_.y) - ray.origin_.y) * inverse_direction.y;
		const double tymax = ((ray.direction_.y < 0 ? min_.y : max_.y) - ray.origin_.y) * inverse_direction.y;

		if (txmin > tymax || tymin > txmax)
			return false;
		if (tymin > txmin)
			txmin = tymin;
		if (tymax < txmax)
			txmax = tymax;

		const double tzmin = ((ray.direction_.z < 0 ? max_.z : min_.z) - ray.origin_.z) * inverse_direction.z;
		const double tzmax = ((ray.direction_.z < 0 ? min_.z : max_.z) - ray.origin_.z) * inverse_direction.z;

		if (txmin > tzmax || tzmin > txmax)
			return false;
		if (tzmin > txmin)
			txmin = tzmin;
		if (tzmax < txmax)
			txmax = tzmax;
		return txmin < closest_t && txmax > std::numeric_limits<double>::epsilon();
	}

	[[nodiscard]] glm::dvec3 GetMin() const { return min_; }
	[[nodiscard]] glm::dvec3 GetMax() const { return max_; }

	void enclose(const AABB& other)
	{
		this->min_.x = glm::min(this->min_.x, other.min_.x);
		this->min_.y = glm::min(this->min_.y, other.min_.y);
		this->min_.z = glm::min(this->min_.z, other.min_.z);

		this->max_.x = glm::max(this->max_.x, other.max_.x);
		this->max_.y = glm::max(this->max_.y, other.max_.y);
		this->max_.z = glm::max(this->max_.z, other.max_.z);
	}

	void enclose(const AABB& first, const AABB& second)
	{
		this->min_.x = glm::min(first.min_.x, second.min_.x);
		this->min_.y = glm::min(first.min_.y, second.min_.y);
		this->min_.z = glm::min(first.min_.z, second.min_.z);

		this->max_.x = glm::max(first.max_.x, second.max_.x);
		this->max_.y = glm::max(first.max_.y, second.max_.y);
		this->max_.z = glm::max(first.max_.z, second.max_.z);
	}

	void enclose(const glm::dvec3& point)
	{
		this->min_.x = glm::min(this->min_.x, point.x);
		this->min_.y = glm::min(this->min_.y, point.y);
		this->min_.z = glm::min(this->min_.z, point.z);

		this->max_.x = glm::max(this->max_.x, point.x);
		this->max_.y = glm::max(this->max_.y, point.y);
		this->max_.z = glm::max(this->max_.z, point.z);
	}

private:
	glm::dvec3 min_;
	glm::dvec3 max_;
};