#pragma once
#include "App.hpp"

class Image
{
public:
	Image(uint32_t width, uint32_t height, const void* data = nullptr);
	Image(const Image&) = delete;
	Image(Image&&) = delete;
	Image operator= (const Image&) = delete;
	Image operator= (Image&&) = delete;
	~Image();

	void SetData(const void* data);

	[[nodiscard]] VkDescriptorSet GetDescriptorSet() const { return descriptor_set_; }

	[[nodiscard]] uint32_t GetWidth() const { return width_; }
	[[nodiscard]] uint32_t GetHeight() const { return height_; }

private:
	void AllocateMemory();
	void ReleaseMemory();

	uint32_t width_ = 0, height_ = 0;

	VkImage image_ = nullptr;
	VkImageView image_view_ = nullptr;
	VkDeviceMemory memory_ = nullptr;
	VkSampler sampler_ = nullptr;
	VkBuffer staging_buffer_ = nullptr;
	VkDeviceMemory staging_buffer_memory_ = nullptr;

	size_t required_memory_size_ = 0;

	VkDescriptorSet descriptor_set_ = nullptr;
};