#pragma once
#include "Unions.hpp"

#include "stb_image.h"

#include <cstdint>
#include <string>
#include <utility>

enum TextureType
{
	UNKNOWN_TEXTURE,
	SOLID,
	IMAGE,
	CHECKER
};

struct TextureInfo
{
	explicit TextureInfo(const TextureType type, std::string texture_name)
		: type(type), name(std::move(texture_name)) {}
	virtual ~TextureInfo() = default;

	TextureInfo(const TextureInfo&) = delete;
	TextureInfo(TextureInfo&&) = default;
	TextureInfo& operator=(const TextureInfo&) = delete;
	TextureInfo& operator=(TextureInfo&&) = default;

	TextureType type{UNKNOWN_TEXTURE};
	std::string name{};
};

struct SolidInfo final : TextureInfo
{
	explicit SolidInfo(const float3 albedo, std::string texture_name)
		: TextureInfo(SOLID, std::move(texture_name)), albedo{albedo} {}

	Float3 albedo{};
};

struct ImageInfo final : TextureInfo
{
	explicit ImageInfo(float* data, const int32_t width, const int32_t height, std::string texture_name)
		: TextureInfo(IMAGE, std::move(texture_name)), buffered_data(data), width(width), height(height) {}

	~ImageInfo() override
	{
		stbi_image_free(buffered_data);
	}

	ImageInfo(const ImageInfo&) = delete;
	ImageInfo(ImageInfo&&) = default;
	ImageInfo& operator=(const ImageInfo&) = delete;
	ImageInfo& operator=(ImageInfo&&) = default;

	float* buffered_data = nullptr;
	float* usable_data = nullptr;
	int32_t width{}, height{};
};

struct CheckerInfo final : TextureInfo
{
	explicit CheckerInfo(const float3 albedo_a, const float3 albedo_b, const float tile_size, std::string texture_name)
		: TextureInfo(CHECKER, std::move(texture_name)), albedo_a{albedo_a}, albedo_b{albedo_b}, tile_size(tile_size) {}

	Float3 albedo_a{};
	Float3 albedo_b{};
	float tile_size{};
};