#pragma once
#include "Intersection.hpp"

#include <vector>

class Scene
{
	class BoundInfo
	{
	public:
		explicit BoundInfo(const std::shared_ptr<Object>& object) : object_(object), aabb_(object->getAABB()),
		                                                            centroid_((aabb_.GetMin() + aabb_.GetMax()) / 2.0)
		{
		}

		[[nodiscard]] glm::vec3 GetCentroid() const { return centroid_; }
		[[nodiscard]] std::shared_ptr<Object> GetObject() const { return object_; }
		[[nodiscard]] AABB getAABB() const { return aabb_; }

	private:
		std::shared_ptr<Object> object_;
		AABB aabb_;
		glm::vec3 centroid_;

	};

	class CentroidComparator
	{
	public:
		explicit CentroidComparator(const int32_t dimension) : dimension_(dimension)
		{
		}

		bool operator()(const BoundInfo& first, const BoundInfo& second) const
		{
			return first.GetCentroid()[dimension_] < second.GetCentroid()[dimension_];
		}

	private:
		int32_t dimension_;
	};

	class BVHNode
	{
	public:
		AABB aabb_{};
		union 
		{
			uint32_t first_object_index;
			uint32_t second_child_index;
		};
		uint8_t object_count_;
		uint8_t split_axis_;
	};

public:
	[[nodiscard]] Intersection intersect(const Ray& ray) const;
 	void RebuildBvh(uint8_t max_objects_per_leaf);
	void add(std::shared_ptr<Object> object, const std::shared_ptr<Material>& material);

private:
	void SplitBoundsRecursively(std::vector<BoundInfo>::iterator begin, std::vector<BoundInfo>::iterator end, uint8_t max_objects_per_leaf);

	std::vector<std::shared_ptr<Object>> bounded_objects_;
	std::vector<std::shared_ptr<Object>> unbounded_objects_;
	std::vector<BoundInfo> bound_infos_;
	std::vector<BVHNode> bvh_;
};