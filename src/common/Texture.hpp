#pragma once
#include "../common/Math.hpp"

#include <cstdint>

enum class TextureType
{
	UNKNOWN_TEXTURE,
	SOLID,
	IMAGE,
	CHECKER
};

struct Solid
{
	__host__ Solid(const float3 albedo)
		: albedo(albedo) {}

	__inline__ __host__ __device__ float3 color(const float2) const
	{
		return albedo;
	}

	float3 albedo{};
};

struct Image
{
	__host__ Image(float* h_data, const int32_t width, const int32_t height)
		: h_data(h_data), width(width), height(height) {}

	__inline__ __host__ __device__ float3 color(const float2 uv) const
	{
		const auto i = (int32_t)(uv.x * (float)width);
		const auto j = (int32_t)(uv.y * (float)height);

		const int32_t texel_index = 3 * (j * width + i);

		if (texel_index < 0 || texel_index > 3 * width * height + 2)
			return make_float3(0.0f);
		

		return make_float3(d_data[texel_index], d_data[texel_index + 1], d_data[texel_index + 2]);
	}

	float* h_data = nullptr;
	float* d_data = nullptr;
	int32_t width{}, height{};
};

struct Checker
{
	__host__ Checker(const float3 albedo_a, const float3 albedo_b, const float tile_size)
		: albedo_a(albedo_a), albedo_b(albedo_b), tile_size(tile_size) {}

	__inline__ __host__ __device__ float3 color(const float2 uv) const
	{
		return sin(uv.x / tile_size) * sin(uv.y / tile_size) < 0.0f ? albedo_a : albedo_b;
	}

	float3 albedo_a{};
	float3 albedo_b{};
	float tile_size{};
};

struct Texture
{
	__host__ __device__ Texture() {}

	explicit __host__ Texture(Solid&& texture)
		: type(TextureType::SOLID), solid(texture) {}

	explicit __host__ Texture(Image&& texture)
		: type(TextureType::IMAGE), image(texture) {}

	explicit __host__ Texture(Checker&& texture)
		: type(TextureType::CHECKER), checker(texture) {}

	__inline__ __host__ __device__ float3 color(const float2 uv) const
	{
		if (type == TextureType::SOLID)
			return solid.color(uv);
		if (type == TextureType::IMAGE)
			return image.color(uv);
		if (type == TextureType::CHECKER)
			return checker.color(uv);
		return {};
	}

	TextureType type{TextureType::UNKNOWN_TEXTURE};

	union
	{
		Solid solid;
		Image image;
		Checker checker;
	};
};