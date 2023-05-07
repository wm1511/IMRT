#pragma once
#include <cstdint>
#include <string>
#include <utility>

enum class MaterialType
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

	[[nodiscard]] virtual uint64_t get_size() const = 0;

	MaterialType type{MaterialType::UNKNOWN_MATERIAL};
	int32_t texture_id{0};
	std::string name{};
};

struct DiffuseInfo final : MaterialInfo
{
	explicit DiffuseInfo(const int32_t texture_info, std::string material_name)
		: MaterialInfo(MaterialType::DIFFUSE, texture_info, std::move(material_name)) {}

	[[nodiscard]] uint64_t get_size() const override
	{
		return sizeof(DiffuseInfo);
	}
};

struct SpecularInfo final : MaterialInfo
{
	SpecularInfo(const float fuzziness, const int32_t texture_info, std::string material_name)
		: MaterialInfo(MaterialType::SPECULAR, texture_info, std::move(material_name)), fuzziness(fuzziness) {}

	[[nodiscard]] uint64_t get_size() const override
	{
		return sizeof(SpecularInfo);
	}

	float fuzziness{};
};

struct RefractiveInfo final : MaterialInfo
{
	explicit RefractiveInfo(const float refractive_index, std::string material_name)
		: MaterialInfo(MaterialType::REFRACTIVE, NULL, std::move(material_name)), refractive_index(refractive_index) {}

	[[nodiscard]] uint64_t get_size() const override
	{
		return sizeof(RefractiveInfo);
	}

	float refractive_index{};
};

struct IsotropicInfo final : MaterialInfo
{
	explicit IsotropicInfo(const int32_t texture_info, std::string material_name)
		: MaterialInfo(MaterialType::ISOTROPIC, texture_info, std::move(material_name)) {}

	[[nodiscard]] uint64_t get_size() const override
	{
		return sizeof(IsotropicInfo);
	}
};