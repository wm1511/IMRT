#pragma once
#include "Unions.hpp"

#include "stb_image.h"

#include <cstdint>

enum TextureType
{
	UNKNOWN_TEXTURE,
	SOLID,
	IMAGE,
	CHECKER
};

struct TextureInfo
{
	TextureInfo() = default;
	explicit TextureInfo(const TextureType type) : type(type) {}
	virtual ~TextureInfo() = default;

	TextureInfo(const TextureInfo&) = delete;
	TextureInfo(TextureInfo&&) = default;
	TextureInfo& operator=(const TextureInfo&) = delete;
	TextureInfo& operator=(TextureInfo&&) = default;

	TextureType type{UNKNOWN_TEXTURE};
};

struct SolidInfo final : TextureInfo
{
	SolidInfo() = default;
	explicit SolidInfo(const float3 albedo)
		: TextureInfo(SOLID), albedo{albedo} {}

	Float3 albedo{};
};

struct ImageInfo final : TextureInfo
{
	ImageInfo() = default;
	explicit ImageInfo(float* data, const int32_t width, const int32_t height)
		: TextureInfo(IMAGE), buffered_data(data), width(width), height(height) {}

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
	CheckerInfo() = default;
	explicit CheckerInfo(const float3 albedo_a, const float3 albedo_b, const float tile_size)
		: TextureInfo(CHECKER), albedo_a{albedo_a}, albedo_b{albedo_b}, tile_size(tile_size) {}

	Float3 albedo_a{};
	Float3 albedo_b{};
	float tile_size{};
};