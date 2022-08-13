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

#include "Scene.hpp"

#include "Sphere.hpp"
#include "Plane.hpp"
#include "Diffuse.hpp"
#include "Specular.hpp"
#include "Refractive.hpp"

#include <algorithm>

Intersection Scene::intersect(const Ray& ray) const
{
    Intersection closest;

    for (const auto& object : mUnboundedObjects)
    {
	    const float t = object->intersect(ray);
        if (t > std::numeric_limits<float>::epsilon() && t < closest.getT())
        {
	        closest.setT(t);
            closest.setObject(object);
        }
    }

    if (!mBVH.empty())
    {
	    const glm::vec3 inverseDirection{1.0f / ray.getDirection()};

        uint32_t stack[64];
        uint32_t nodeNumber = 0, stackSize = 0;

        while (true)
        {
	        const BVHNode& node = mBVH[nodeNumber];
            if(node.aabb.intersect(ray, inverseDirection, closest.getT()))
            {
	            if (node.objectCount > 0)
	            {
		            for (uint64_t objectNumber = 0; objectNumber != node.objectCount; ++objectNumber)
		            {
			            const float t = mBoundedObjects[node.firstObjectIndex + objectNumber]->intersect(ray);
                        if (t > std::numeric_limits<float>::epsilon() && t < closest.getT())
                        {
	                        closest.setT(t);
                            closest.setObject(mBoundedObjects[node.firstObjectIndex + objectNumber]);
                        }
		            }
                    if (stackSize == 0)
                        break;
                    nodeNumber = stack[--stackSize];
	            }
                else
                {
	                if (ray.getDirection()[node.splitAxis] < 0)
	                {
		                stack[stackSize++] = nodeNumber + 1;
                        nodeNumber = node.secondChildIndex;
	                }
                    else
                    {
	                    stack[stackSize++] = node.secondChildIndex;
                        nodeNumber += 1;
                    }
                }
            }
            else
            {
	            if (stackSize == 0)
                    break;
            	nodeNumber = stack[--stackSize];
            }
        }
    }

    return closest;
}

void Scene::add(std::shared_ptr<Object> object, const std::shared_ptr<Material>& material)
{
    object->setMaterial(material);
    if (object->getAABB().unbounded())
		mUnboundedObjects.push_back(std::move(object));
    else
        mBoundInfos.emplace_back(object);
}

void Scene::rebuildBVH(uint8_t maxObjectsPerLeaf)
{
    if (maxObjectsPerLeaf == 0)
        maxObjectsPerLeaf = 1;

    mBVH.clear();
    mBVH.reserve(4 * mBoundInfos.size());
    splitBoundsRecursively(mBoundInfos.begin(), mBoundInfos.end(), maxObjectsPerLeaf);

    mBoundedObjects.clear();
    mBoundedObjects.reserve(mBoundInfos.size());
    for (auto& boundInfo : mBoundInfos)
        mBoundedObjects.emplace_back(std::move(boundInfo.getObject()));
}

void Scene::splitBoundsRecursively(const std::vector<BoundInfo>::iterator begin,
	const std::vector<BoundInfo>::iterator end, uint8_t maxObjectsPerLeaf)
{
    mBVH.emplace_back(BVHNode());
    auto thisNode = mBVH.end();
    --thisNode;

    const uint64_t objectCount = end - begin;
    if (objectCount <= maxObjectsPerLeaf)
    {
	    thisNode->objectCount = static_cast<uint8_t>(objectCount);
        for (auto it = begin; it != end; ++it)
            thisNode->aabb.enclose(it->getAABB());
        thisNode->firstObjectIndex = static_cast<uint32_t>(begin - mBoundInfos.begin());
    }
    else
    {
	    thisNode->objectCount = 0;

        AABB centroidBound;
        for (auto it = begin; it != end; ++it)
            centroidBound.enclose(it->getAABB());

        thisNode->splitAxis = static_cast<uint8_t>(centroidBound.getLargestDimension());
        const auto medianIterator = begin + (end - begin) / 2;
        std::nth_element(begin, medianIterator, end, CentroidComparator(thisNode->splitAxis));

        const uint64_t firstChildIndex = mBVH.size();
        splitBoundsRecursively(begin, medianIterator, maxObjectsPerLeaf);
        thisNode->secondChildIndex = static_cast<uint32_t>(mBVH.size());
        splitBoundsRecursively(medianIterator, end, maxObjectsPerLeaf);

        thisNode->aabb.enclose(mBVH[firstChildIndex].aabb, mBVH[thisNode->secondChildIndex].aabb);
    }
}

Scene Scene::makeCornellBox()
{
    Scene scene;

    const auto specular = std::make_shared<Specular>(glm::vec3(4, 8, 4));
    const auto refractive = std::make_shared<Refractive>(glm::vec3(10, 10, 1), 1.5f);
    const auto blue = std::make_shared<Diffuse>(glm::vec3(4, 4, 12));
    const auto light = std::make_shared<Diffuse>(glm::vec3(0, 0, 0), 10000.0f);
    const auto gray = std::make_shared<Diffuse>(glm::vec3(6, 6, 6));
    const auto red = std::make_shared<Diffuse>(glm::vec3(10, 2, 2));
    const auto green = std::make_shared<Diffuse>(glm::vec3(2, 10, 2));

	scene.add(std::make_shared<Sphere>(glm::vec3(-0.75, -1.45, -4.4), 1.05f), specular);
    scene.add(std::make_shared<Sphere>(glm::vec3(2.0, -2.05, -3.7), 0.5f), refractive);
    scene.add(std::make_shared<Sphere>(glm::vec3(-1.75, -1.95, -3.1), 0.6f), blue);
    scene.add(std::make_shared<Sphere>(glm::vec3(0, 1.9, -3), 0.5f), light);

    scene.add(std::make_shared<Plane>(glm::vec3(1, 0, 0), 2.75f), red);
    scene.add(std::make_shared<Plane>(glm::vec3(-1, 0, 0), 2.75f), green);
    scene.add(std::make_shared<Plane>(glm::vec3(0, 1, 0), 2.5f), gray);
    scene.add(std::make_shared<Plane>(glm::vec3(0, -1, 0), 3.0f), gray);
    scene.add(std::make_shared<Plane>(glm::vec3(0, 0, 1), 5.5f), gray);
    scene.add(std::make_shared<Plane>(glm::vec3(0, 0, -1), 0.5f), gray);

    return scene;
}