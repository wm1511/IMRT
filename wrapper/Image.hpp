#pragma once
#include "App.hpp"

class Image
{
public:
	Image(uint32_t width, uint32_t height, const void* data = nullptr);
	Image(const Image&) = delete;
	Image(Image&&) = delete;
	Image& operator= (const Image&) = delete;
	Image& operator= (Image&&) = delete;
	~Image();

	void set_data(const void* data);

	[[nodiscard]] VkDescriptorSet get_descriptor_set() const { return descriptor_set_; }

	[[nodiscard]] uint32_t get_width() const { return width_; }
	[[nodiscard]] uint32_t get_height() const { return height_; }

private:
	void allocate_memory();
	void release_memory();

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