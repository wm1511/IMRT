#pragma once
#include "Unions.hpp"

#include <cstdint>

enum ObjectType
{
	UNKNOWN_OBJECT,
	SPHERE,
	TRIANGLE,
	PLANE,
	VOLUMETRIC_SPHERE,
	CYLINDER,
	CONE,
	TORUS,
	MODEL
};

struct ObjectInfo
{
	ObjectInfo() = default;
	ObjectInfo(const ObjectType type, const int32_t material_info) : type(type), material_id(material_info) {}
	virtual ~ObjectInfo() = default;

	ObjectInfo(const ObjectInfo&) = delete;
	ObjectInfo(ObjectInfo&&) = default;
	ObjectInfo& operator=(const ObjectInfo&) = delete;
	ObjectInfo& operator=(ObjectInfo&&) = default;

	ObjectType type{UNKNOWN_OBJECT};
	int32_t material_id{0};
};

struct SphereInfo final : ObjectInfo
{
	SphereInfo() = default;
	SphereInfo(const float3 center, const float radius, const int32_t material_info)
		: ObjectInfo(SPHERE, material_info), center{center}, radius(radius) {}

	Float3 center{};
	float radius{};
};

struct TriangleInfo final : ObjectInfo
{
	TriangleInfo() = default;
	TriangleInfo(const float3 v0, const float3 v1, const float3 v2, const int32_t material_info, const float3 normal, const float2 min_uv, const float2 max_uv)
		: ObjectInfo(TRIANGLE, material_info), v0{v0}, v1{v1}, v2{v2}, normal(normal), min_uv(min_uv), max_uv(max_uv) {}

	Float3 v0{}, v1{}, v2{};
	float3 normal{};
	float2 min_uv{}, max_uv{};
};

struct PlaneInfo final : ObjectInfo
{
	PlaneInfo() = default;
	PlaneInfo(const float3 normal, const float offset, const int32_t material_info)
		: ObjectInfo(PLANE, material_info), normal{normal}, offset(offset) {}

	Float3 normal{};
	float offset{};
};

struct VolumetricSphereInfo final : ObjectInfo
{
	VolumetricSphereInfo() = default;
	VolumetricSphereInfo(const float3 center, const float radius, const float density, const int32_t material_info)
		: ObjectInfo(VOLUMETRIC_SPHERE, material_info), boundary(center, radius, material_info), density(density) {}

	SphereInfo boundary{};
	float density{};
};

struct CylinderInfo final : ObjectInfo
{
	CylinderInfo() = default;
	CylinderInfo(const float3 extreme_a, const float3 extreme_b, const float3 center, const float radius, const int32_t material_info)
		: ObjectInfo(CYLINDER, material_info), extreme_a{extreme_a}, extreme_b{extreme_b}, center{center}, radius(radius) {}

	Float3 extreme_a{}, extreme_b{}, center{};
	float radius{};
};

struct ConeInfo final : ObjectInfo
{
	ConeInfo() = default;
	ConeInfo(const float3 extreme_a, const float3 extreme_b, const float3 center, const float radius_a, const float radius_b, const int32_t material_info)
		: ObjectInfo(CONE, material_info), extreme_a{extreme_a}, extreme_b{extreme_b}, center{center}, radius_a(radius_a), radius_b(radius_b) {}

	Float3 extreme_a{}, extreme_b{}, center{};
	float radius_a{}, radius_b{};
};

struct TorusInfo final : ObjectInfo
{
	TorusInfo() = default;
	TorusInfo(const float3 center, const float radius_a, const float radius_b, const int32_t material_info)
		: ObjectInfo(TORUS, material_info), center{center}, radius_a(radius_a), radius_b(radius_b) {}

	Float3 center{};
	float radius_a{}, radius_b{};
};

struct ModelInfo final : ObjectInfo
{
	ModelInfo() = default;
	ModelInfo(TriangleInfo* triangles, const uint64_t triangle_count, const int32_t material_info)
		: ObjectInfo(MODEL, material_info), buffered_triangles(triangles), triangle_count(triangle_count) {}

	~ModelInfo() override
	{
		delete[] buffered_triangles;
	}

	ModelInfo(const ModelInfo&) = delete;
	ModelInfo(ModelInfo&&) = delete;
	ModelInfo operator=(const ModelInfo&) = delete;
	ModelInfo operator=(ModelInfo&&) = delete;

	Float3 translation{{0.0f, 0.0f, 0.0f}};
	Float3 scale{{1.0f, 1.0f, 1.0f}};
	Float3 rotation{{0.0f, 0.0f, 0.0f}};
	TriangleInfo* buffered_triangles = nullptr;
	TriangleInfo* usable_triangles = nullptr;
	uint64_t triangle_count{};
};