#pragma once
#include "MaterialInfo.hpp"
#include "TransformInfo.hpp"

#include <memory>

enum ObjectType
{
	UNKNOWN_OBJECT,
	SPHERE,
	TRIANGLE,
	TRIANGLE_MESH
};

struct ObjectInfo
{
	ObjectType object_type = UNKNOWN_OBJECT;
	std::shared_ptr<MaterialInfo> material = nullptr;
	TransformInfo transform{{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f} };
};

struct SphereInfo : ObjectInfo
{
	SphereInfo(float center[3], const float radius, const std::shared_ptr<MaterialInfo>& material_info) : center{center[0], center[1], center[2]}, radius(radius)
	{
		object_type = SPHERE;
		material = material_info;
	}
	
	float center[3];
	float radius;
};

struct TriangleInfo : ObjectInfo
{
	TriangleInfo(float v0[3], float v1[3], float v2[3], const std::shared_ptr<MaterialInfo>& material_info) : v0{v0[0], v0[1], v0[2]}, v1{v1[0], v1[1], v1[2]}, v2{v2[0], v2[1], v2[2]}
	{
		object_type = TRIANGLE;
		material = material_info;
	}

	float v0[3], v1[3], v2[3];
};