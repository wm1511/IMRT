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
	SphereInfo(float3 center, const float radius, const uint32_t material_info) : center{center}, radius(radius)
	{
		type = SPHERE;
		material_id = material_info;
	}

	union
	{
		float3 center;
		float center_array[3];
	};
	float radius;
};

struct TriangleInfo : ObjectInfo
{
	TriangleInfo(float3 v0, float3 v1, float3 v2, const uint32_t material_info) : v0{v0}, v1{v1}, v2{v2}
	{
		type = TRIANGLE;
		material_id = material_info;
	}

	union
	{
		float3 v0;
		float v0_array[3];
	};
	union
	{
		float3 v1;
		float v1_array[3];
	};
	union
	{
		float3 v2;
		float v2_array[3];
	};
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