#pragma once
#include "TransformInfo.hpp"

#include <cstdint>

enum ObjectType
{
	UNKNOWN_OBJECT,
	SPHERE,
	TRIANGLE,
	TRIANGLE_MESH
};

struct ObjectInfo
{
	ObjectType type{UNKNOWN_OBJECT};
	uint32_t material_id{0};
};

struct SphereInfo : ObjectInfo
{
	SphereInfo(float center[3], const float radius, const uint32_t material_info) : center{center[0], center[1], center[2]}, radius(radius)
	{
		type = SPHERE;
		material_id = material_info;
	}
	
	float center[3];
	float radius;
};

struct TriangleInfo : ObjectInfo
{
	TriangleInfo(float v0[3], float v1[3], float v2[3], const uint32_t material_info) : v0{v0[0], v0[1], v0[2]}, v1{v1[0], v1[1], v1[2]}, v2{v2[0], v2[1], v2[2]}
	{
		type = TRIANGLE;
		material_id = material_info;
	}

	float v0[3], v1[3], v2[3];
};

struct TriangleMeshInfo : ObjectInfo
{
	TriangleMeshInfo(const uint32_t material_info)
	{
		type = TRIANGLE_MESH;
		material_id = material_info;
	}

	TriangleInfo* triangle_list = nullptr;
	uint32_t triangle_count{0};
	TransformInfo transform{{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f}};
};