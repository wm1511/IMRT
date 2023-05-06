#pragma once
#include "Math.cuh"
#include "../info/TextureInfo.hpp"

class Texture
{
public:
	__host__ __device__ Texture() {}
	__host__ __device__ virtual ~Texture() {}

	__host__ __device__ Texture(const Texture&) = delete;
	__host__ __device__ Texture(Texture&&) = delete;
	__host__ __device__ Texture& operator=(const Texture&) = delete;
	__host__ __device__ Texture& operator=(Texture&&) = delete;

	__host__ __device__ [[nodiscard]] virtual float3 color(float2 uv) const = 0;
	__host__ __device__ virtual void update(TextureInfo* texture_info) = 0;
};

class Solid final : public Texture
{
public:
	__host__ __device__ explicit Solid(const SolidInfo* solid_info)
		: albedo_(solid_info->albedo.str) {}

	__host__ __device__ [[nodiscard]] float3 color(const float2) const override
	{
		return albedo_;
	}

	__host__ __device__ void update(TextureInfo* texture) override
	{
		const SolidInfo* solid_info = (SolidInfo*)texture;
		albedo_ = solid_info->albedo.str;
	}

private:
	float3 albedo_{};
};

class Image final : public Texture
{
public:
	__host__ __device__ explicit Image(const ImageInfo* image_info)
		: data_(image_info->usable_data), width_(image_info->width), height_(image_info->height) {}

	__host__ __device__ [[nodiscard]] float3 color(const float2 uv) const override
	{
		const auto i = (int32_t)(uv.x * (float)width_);
		const auto j = (int32_t)(uv.y * (float)height_);

		const int32_t texel_index = 3 * (j * width_ + i);

		if (texel_index < 0 || texel_index > 3 * width_ * height_ + 2)
			return make_float3(0.0f);
		

		return make_float3(data_[texel_index], data_[texel_index + 1], data_[texel_index + 2]);
	}

	__host__ __device__ void update(TextureInfo* texture) override
	{
		const ImageInfo* image_info = (ImageInfo*)texture;
		data_ = image_info->usable_data;
		width_ = image_info->width;
		height_ = image_info->height;
	}

private:
	float* data_ = nullptr;
	int32_t width_{}, height_{};
};

class Checker final : public Texture
{
public:
	__host__ __device__ explicit Checker(const CheckerInfo* checker_info)
		: albedo_a_(checker_info->albedo_a.str), albedo_b_(checker_info->albedo_b.str), inverse_size_(1.0f / checker_info->tile_size) {}

	__host__ __device__ [[nodiscard]] float3 color(const float2 uv) const override
	{
		return sin(inverse_size_ * uv.x) * sin(inverse_size_ * uv.y) < 0.0f ? albedo_a_ : albedo_b_;
	}

	__host__ __device__ void update(TextureInfo* texture) override
	{
		const CheckerInfo* checker_info = (CheckerInfo*)texture;
		albedo_a_ = checker_info->albedo_a.str;
		albedo_b_ = checker_info->albedo_b.str;
		inverse_size_ = 1.0f / checker_info->tile_size;
	}

private:
	float3 albedo_a_{};
	float3 albedo_b_{};
	float inverse_size_{};
};