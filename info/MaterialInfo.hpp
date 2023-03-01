#pragma once
#include "Unions.hpp"

#include "stb_image.h"

enum MaterialType
{
	UNKNOWN_MATERIAL,
	DIFFUSE,
	SPECULAR,
	REFRACTIVE,
	ISOTROPIC,
	TEXTURE
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
	explicit DiffuseInfo(const float3 albedo)
		: MaterialInfo(DIFFUSE), albedo{albedo} {}

	Float3 albedo{};
};

struct SpecularInfo final : MaterialInfo
{
	SpecularInfo() = default;
	SpecularInfo(const float3 albedo, const float fuzziness)
		: MaterialInfo(SPECULAR), albedo{albedo}, fuzziness(fuzziness) {}

	Float3 albedo{};
	float fuzziness{};
};

struct RefractiveInfo final : MaterialInfo
{
	RefractiveInfo() = default;
	explicit RefractiveInfo(const float refractive_index)
		: MaterialInfo(REFRACTIVE), refractive_index(refractive_index) {}

	float refractive_index{};
};

struct IsotropicInfo final : MaterialInfo
{
	IsotropicInfo() = default;
	explicit IsotropicInfo(const float3 albedo)
		: MaterialInfo(ISOTROPIC), albedo{albedo} {}

	Float3 albedo{};
};

struct TextureInfo final : MaterialInfo
{
	TextureInfo() = default;
	explicit TextureInfo(float* data, const int32_t width, const int32_t height)
		: MaterialInfo(TEXTURE), buffered_data(data), width(width), height(height) {}

	~TextureInfo() override
	{
		stbi_image_free(buffered_data);
	}

	TextureInfo(const TextureInfo&) = delete;
	TextureInfo(TextureInfo&&) = default;
	TextureInfo& operator=(const TextureInfo&) = delete;
	TextureInfo& operator=(TextureInfo&&) = default;

	float* buffered_data = nullptr;
	float* usable_data = nullptr;
	int32_t width{}, height{};
};