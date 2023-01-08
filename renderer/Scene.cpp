#include "Scene.hpp"

#include <algorithm>

Intersection Scene::intersect(const Ray& ray) const
{
    Intersection closest;

    for (const auto& object : unbounded_objects_)
    {
	    const double t = object->intersect(ray);
        if (t > std::numeric_limits<double>::epsilon() && t < closest.t_)
        {
	        closest.t_ = t;
            closest.object_ = object;
        }
    }

    if (!bvh_.empty())
    {
        const glm::vec3 inverse_direction{1.0 / ray.direction_};

        uint32_t stack[64];
        uint32_t node_number = 0, stack_size = 0;

        while (true)
        {
	        const BVHNode& node = bvh_[node_number];
            if (node.aabb_.intersect(ray, inverse_direction, closest.t_))
            {
	            if (node.object_count_ > 0)
	            {
		            for (uint64_t object_number = 0; object_number != node.object_count_; ++object_number)
		            {
			            const double t = bounded_objects_[node.first_object_index + object_number]->intersect(ray);
                        if (t > std::numeric_limits<double>::epsilon() && t < closest.t_)
                        {
	                        closest.t_ = t;
                            closest.object_ = bounded_objects_[node.first_object_index + object_number];
                        }
		            }
                    if (stack_size == 0)
                        break;
                    node_number = stack[--stack_size];
	            }
                else
                {
	                if (ray.direction_[node.split_axis_] < 0)
	                {
		                stack[stack_size++] = node_number + 1;
                        node_number = node.second_child_index;
	                }
                    else
                    {
	                    stack[stack_size++] = node.second_child_index;
                        node_number += 1;
                    }
                }
            }
            else
            {
	            if (stack_size == 0)
                    break;
            	node_number = stack[--stack_size];
            }
        }
    }

    return closest;
}

void Scene::add(std::shared_ptr<Object> object, const std::shared_ptr<Material>& material)
{
    object->material_ = material;
    if (object->getAABB().unbounded())
		unbounded_objects_.push_back(std::move(object));
    else
        bound_infos_.emplace_back(object);
}

void Scene::RebuildBvh(uint8_t max_objects_per_leaf)
{
    if (max_objects_per_leaf == 0)
        max_objects_per_leaf = 1;

    bvh_.clear();
    bvh_.reserve(4 * bound_infos_.size());
    SplitBoundsRecursively(bound_infos_.begin(), bound_infos_.end(), max_objects_per_leaf);

    bounded_objects_.clear();
    bounded_objects_.reserve(bound_infos_.size());
    for (auto& bound_info : bound_infos_)
        bounded_objects_.emplace_back(bound_info.GetObject());
}

void Scene::SplitBoundsRecursively(const std::vector<BoundInfo>::iterator begin,
	const std::vector<BoundInfo>::iterator end, uint8_t max_objects_per_leaf)
{
    bvh_.emplace_back(BVHNode());
    auto this_node = bvh_.end();
    --this_node;

    const uint64_t object_count = end - begin;
    if (object_count <= max_objects_per_leaf)
    {
	    this_node->object_count_ = static_cast<uint8_t>(object_count);
        for (auto it = begin; it != end; ++it)
            this_node->aabb_.enclose(it->getAABB());
        this_node->first_object_index = static_cast<uint32_t>(begin - bound_infos_.begin());
    }
    else
    {
	    this_node->object_count_ = 0;

        AABB centroid_bound;
        for (auto it = begin; it != end; ++it)
            centroid_bound.enclose(it->getAABB());

        this_node->split_axis_ = static_cast<uint8_t>(centroid_bound.GetLargestDimension());
        const auto median_iterator = begin + (end - begin) / 2;
        std::nth_element(begin, median_iterator, end, CentroidComparator(this_node->split_axis_));

        const uint64_t first_child_index = bvh_.size();
        SplitBoundsRecursively(begin, median_iterator, max_objects_per_leaf);
        this_node->second_child_index = static_cast<uint32_t>(bvh_.size());
        SplitBoundsRecursively(median_iterator, end, max_objects_per_leaf);

        this_node->aabb_.enclose(bvh_[first_child_index].aabb_, bvh_[this_node->second_child_index].aabb_);
    }
}