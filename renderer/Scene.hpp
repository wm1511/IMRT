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
#include "Intersection.hpp"

#include <vector>

class Scene
{
	class BoundInfo
	{
	public:
		explicit BoundInfo(const std::shared_ptr<Object>& object) : mObject(object), mAABB(object->getAABB()),
		                                                            mCentroid((mAABB.getMin() + mAABB.getMax()) / 2.0f)
		{
		}

		[[nodiscard]] glm::vec3 getCentroid() const { return mCentroid; }
		[[nodiscard]] std::shared_ptr<Object> getObject() const { return mObject; }
		[[nodiscard]] AABB getAABB() const { return mAABB; }

	private:
		std::shared_ptr<Object> mObject;
		AABB mAABB;
		glm::vec3 mCentroid;

	};

	class CentroidComparator
	{
	public:
		explicit CentroidComparator(const int32_t dimension) : mDimension(dimension)
		{
		}

		bool operator()(const BoundInfo& first, const BoundInfo& second) const
		{
			return first.getCentroid()[mDimension] < second.getCentroid()[mDimension];
		}

	private:
		int32_t mDimension;
	};

	class BVHNode
	{
	public:
		AABB aabb{};
		union 
		{
			uint32_t firstObjectIndex;
			uint32_t secondChildIndex;
		};
		uint8_t objectCount;
		uint8_t splitAxis;
	};

public:
	[[nodiscard]] static Scene makeCornellBox();
	[[nodiscard]] Intersection intersect(const Ray& ray) const;
	void rebuildBVH(uint8_t maxObjectsPerLeaf);

	static constexpr uint32_t SAMPLES_PER_PIXEL = 8;

private:
	Scene() = default;
	void add(std::shared_ptr<Object> object, const std::shared_ptr<Material>& material);
	void splitBoundsRecursively(std::vector<BoundInfo>::iterator begin, std::vector<BoundInfo>::iterator end, uint8_t maxObjectsPerLeaf);

	std::vector<std::shared_ptr<Object>> mBoundedObjects;
	std::vector<std::shared_ptr<Object>> mUnboundedObjects;
	std::vector<BoundInfo> mBoundInfos;
	std::vector<BVHNode> mBVH;
};