#pragma once
#include <cstdint>

enum MaterialType
{
	UNKNOWN_MATERIAL,
	DIFFUSE,
	SPECULAR,
	REFRACTIVE,
	ISOTROPIC
};

struct MaterialInfo
{
	MaterialInfo() = default;
	explicit MaterialInfo(const MaterialType type, const int32_t texture_info) : type(type), texture_id(texture_info) {}
	virtual ~MaterialInfo() = default;

	MaterialInfo(const MaterialInfo&) = delete;
	MaterialInfo(MaterialInfo&&) = default;
	MaterialInfo& operator=(const MaterialInfo&) = delete;
	MaterialInfo& operator=(MaterialInfo&&) = default;

	MaterialType type{UNKNOWN_MATERIAL};
	int32_t texture_id{0};
};

struct DiffuseInfo final : MaterialInfo
{
	DiffuseInfo() = default;
	explicit DiffuseInfo(const int32_t texture_info)
		: MaterialInfo(DIFFUSE, texture_info) {}
};

struct SpecularInfo final : MaterialInfo
{
	SpecularInfo() = default;
	SpecularInfo(const float fuzziness, const int32_t texture_info)
		: MaterialInfo(SPECULAR, texture_info), fuzziness(fuzziness) {}

	float fuzziness{};
};

struct RefractiveInfo final : MaterialInfo
{
	RefractiveInfo() = default;
	explicit RefractiveInfo(const float refractive_index)
		: MaterialInfo(REFRACTIVE, NULL), refractive_index(refractive_index) {}

	float refractive_index{};
};

struct IsotropicInfo final : MaterialInfo
{
	IsotropicInfo() = default;
	explicit IsotropicInfo(const int32_t texture_info)
		: MaterialInfo(ISOTROPIC, texture_info) {}
};