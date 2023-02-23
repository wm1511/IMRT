#pragma once
#include "Unions.hpp"

enum MaterialType
{
	UNKNOWN_MATERIAL,
	DIFFUSE,
	SPECULAR,
	REFRACTIVE
};

struct MaterialInfo
{
	MaterialInfo() = default;
	explicit MaterialInfo(const MaterialType type) : type(type) {}
	virtual ~MaterialInfo() = default;

	MaterialInfo(const MaterialInfo&) = delete;
	MaterialInfo(MaterialInfo&&) = default;
	MaterialInfo& operator=(const MaterialInfo&) = delete;
	MaterialInfo& operator=(MaterialInfo&&) = default;

	MaterialType type{UNKNOWN_MATERIAL};
};

struct DiffuseInfo final : MaterialInfo
{
	DiffuseInfo() = default;
	explicit DiffuseInfo(const float3 albedo) : MaterialInfo(DIFFUSE), albedo{albedo} {}

	Float3 albedo{};
};

struct SpecularInfo final : MaterialInfo
{
	SpecularInfo() = default;
	SpecularInfo(const float3 albedo, const float fuzziness) : MaterialInfo(SPECULAR), albedo{albedo}, fuzziness(fuzziness) {}

	Float3 albedo{};
	float fuzziness{};
};

struct RefractiveInfo final : MaterialInfo
{
	RefractiveInfo() = default;
	explicit RefractiveInfo(const float refractive_index) : MaterialInfo(REFRACTIVE), refractive_index(refractive_index) {}

	float refractive_index{};
};