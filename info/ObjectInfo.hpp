#pragma once
#include "Unions.hpp"

#include <cstdint>

enum class ObjectType
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

	[[nodiscard]] virtual uint64_t get_size() const = 0;

	ObjectType type{ObjectType::UNKNOWN_OBJECT};
	int32_t material_id{0};
	std::string name{};
};

struct SphereInfo final : ObjectInfo
{
	SphereInfo() = default;
	SphereInfo(const float3 center, const float radius, const int32_t material_info, std::string object_name)
		: ObjectInfo(ObjectType::SPHERE, material_info, std::move(object_name)), center{center}, radius(radius) {}

	[[nodiscard]] uint64_t get_size() const override
	{
		return sizeof(SphereInfo);
	}

	Float3 center{};
	float radius{};
};

struct PlaneInfo final : ObjectInfo
{
	PlaneInfo() = default;
	PlaneInfo(const float3 normal, const float offset, const int32_t material_info, std::string object_name)
		: ObjectInfo(ObjectType::PLANE, material_info, std::move(object_name)), normal{normal}, offset(offset) {}

	[[nodiscard]] uint64_t get_size() const override
	{
		return sizeof(PlaneInfo);
	}

	Float3 normal{};
	float offset{};
};

struct CylinderInfo final : ObjectInfo
{
	CylinderInfo() = default;
	CylinderInfo(const float3 extreme_a, const float3 extreme_b, const float radius, const int32_t material_info, std::string object_name)
		: ObjectInfo(ObjectType::CYLINDER, material_info, std::move(object_name)), extreme_a{extreme_a}, extreme_b{extreme_b}, radius(radius) {}

	[[nodiscard]] uint64_t get_size() const override
	{
		return sizeof(CylinderInfo);
	}

	Float3 extreme_a{}, extreme_b{};
	float radius{};
};

struct ConeInfo final : ObjectInfo
{
	ConeInfo() = default;
	ConeInfo(const float3 extreme_a, const float3 extreme_b, const float radius, const int32_t material_info, std::string object_name)
		: ObjectInfo(ObjectType::CONE, material_info, std::move(object_name)), extreme_a{extreme_a}, extreme_b{extreme_b}, radius(radius) {}

	[[nodiscard]] uint64_t get_size() const override
	{
		return sizeof(ConeInfo);
	}

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
	ModelInfo(Vertex* vertices, uint32_t* indices, const uint64_t vertex_count, const uint64_t index_count, const int32_t material_info, std::string object_name)
		: ObjectInfo(ObjectType::MODEL, material_info, std::move(object_name)), h_vertices(vertices), h_indices(indices), vertex_count(vertex_count), index_count(index_count) {}

	~ModelInfo() override
	{
		delete[] h_indices;
		delete[] h_vertices;
	}

	ModelInfo(const ModelInfo&) = delete;
	ModelInfo(ModelInfo&&) = delete;
	ModelInfo operator=(const ModelInfo&) = delete;
	ModelInfo operator=(ModelInfo&&) = delete;

	[[nodiscard]] uint64_t get_size() const override
	{
		return sizeof(ModelInfo);
	}

	Float3 translation{{0.0f, 0.0f, 0.0f}};
	Float3 scale{{1.0f, 1.0f, 1.0f}};
	Float3 rotation{{0.0f, 0.0f, 0.0f}};
	Vertex* h_vertices = nullptr, * d_vertices = nullptr;
	uint32_t* h_indices = nullptr, * d_indices = nullptr;
	uint64_t vertex_count{}, index_count{};
};