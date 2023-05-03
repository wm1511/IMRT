#pragma once
#include <cstdint>
#include <string>
#include <utility>

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
	MaterialInfo(const MaterialType type, const int32_t texture_info, std::string material_name)
		: type(type), texture_id(texture_info), name(std::move(material_name)) {}
	virtual ~MaterialInfo() = default;

	MaterialInfo(const MaterialInfo&) = delete;
	MaterialInfo(MaterialInfo&&) = default;
	MaterialInfo& operator=(const MaterialInfo&) = delete;
	MaterialInfo& operator=(MaterialInfo&&) = default;

	MaterialType type{UNKNOWN_MATERIAL};
	int32_t texture_id{0};
	std::string name{};
};

struct DiffuseInfo final : MaterialInfo
{
	explicit DiffuseInfo(const int32_t texture_info, std::string material_name)
		: MaterialInfo(DIFFUSE, texture_info, std::move(material_name)) {}
};

struct SpecularInfo final : MaterialInfo
{
	SpecularInfo(const float fuzziness, const int32_t texture_info, std::string material_name)
		: MaterialInfo(SPECULAR, texture_info, std::move(material_name)), fuzziness(fuzziness) {}

	float fuzziness{};
};

struct RefractiveInfo final : MaterialInfo
{
	explicit RefractiveInfo(const float refractive_index, std::string material_name)
		: MaterialInfo(REFRACTIVE, NULL, std::move(material_name)), refractive_index(refractive_index) {}

	float refractive_index{};
};

struct IsotropicInfo final : MaterialInfo
{
	explicit IsotropicInfo(const int32_t texture_info, std::string material_name)
		: MaterialInfo(ISOTROPIC, texture_info, std::move(material_name)) {}
};