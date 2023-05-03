#pragma once
#include "Unions.hpp"

#include <cstdint>

enum ObjectType
{
	UNKNOWN_OBJECT,
	SPHERE,
	PLANE,
	CYLINDER,
	CONE,
	MODEL
};

struct ObjectInfo
{
	ObjectInfo() = default;
	ObjectInfo(const ObjectType type, const int32_t material_info, std::string object_name)
		: type(type), material_id(material_info), name(std::move(object_name)) {}
	virtual ~ObjectInfo() = default;

	ObjectInfo(const ObjectInfo&) = delete;
	ObjectInfo(ObjectInfo&&) = default;
	ObjectInfo& operator=(const ObjectInfo&) = delete;
	ObjectInfo& operator=(ObjectInfo&&) = default;

	ObjectType type{UNKNOWN_OBJECT};
	int32_t material_id{0};
	std::string name{};
};

struct SphereInfo final : ObjectInfo
{
	SphereInfo() = default;
	SphereInfo(const float3 center, const float radius, const int32_t material_info, std::string object_name)
		: ObjectInfo(SPHERE, material_info, std::move(object_name)), center{center}, radius(radius) {}

	Float3 center{};
	float radius{};
};

struct PlaneInfo final : ObjectInfo
{
	PlaneInfo() = default;
	PlaneInfo(const float3 normal, const float offset, const int32_t material_info, std::string object_name)
		: ObjectInfo(PLANE, material_info, std::move(object_name)), normal{normal}, offset(offset) {}

	Float3 normal{};
	float offset{};
};

struct CylinderInfo final : ObjectInfo
{
	CylinderInfo() = default;
	CylinderInfo(const float3 extreme_a, const float3 extreme_b, const float radius, const int32_t material_info, std::string object_name)
		: ObjectInfo(CYLINDER, material_info, std::move(object_name)), extreme_a{extreme_a}, extreme_b{extreme_b}, radius(radius) {}

	Float3 extreme_a{}, extreme_b{};
	float radius{};
};

struct ConeInfo final : ObjectInfo
{
	ConeInfo() = default;
	ConeInfo(const float3 extreme_a, const float3 extreme_b, const float radius, const int32_t material_info, std::string object_name)
		: ObjectInfo(CONE, material_info, std::move(object_name)), extreme_a{extreme_a}, extreme_b{extreme_b}, radius(radius) {}

	Float3 extreme_a{}, extreme_b{};
	float radius{};
};

struct Vertex
{
	float3 position;
	float3 normal;
	float2 uv;
};

struct ModelInfo final : ObjectInfo
{
	ModelInfo() = default;
	ModelInfo(Vertex* vertices, const uint64_t triangle_count, const int32_t material_info, std::string object_name)
		: ObjectInfo(MODEL, material_info, std::move(object_name)), buffered_vertices(vertices), triangle_count(triangle_count) {}

	~ModelInfo() override
	{
		delete[] buffered_vertices;
	}

	ModelInfo(const ModelInfo&) = delete;
	ModelInfo(ModelInfo&&) = delete;
	ModelInfo operator=(const ModelInfo&) = delete;
	ModelInfo operator=(ModelInfo&&) = delete;

	Float3 translation{{0.0f, 0.0f, 0.0f}};
	Float3 scale{{1.0f, 1.0f, 1.0f}};
	Float3 rotation{{0.0f, 0.0f, 0.0f}};
	Vertex* buffered_vertices = nullptr;
	Vertex* usable_vertices = nullptr;
	uint64_t triangle_count{};
};