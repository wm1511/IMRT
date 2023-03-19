#pragma once
#include "App.hpp"

class Frame
{
public:
	Frame(uint32_t width, uint32_t height, const void* data = nullptr);
	Frame(const Frame&) = delete;
	Frame(Frame&&) = delete;
	Frame& operator= (const Frame&) = delete;
	Frame& operator= (Frame&&) = delete;
	~Frame();

	void set_data(const void* data);

	[[nodiscard]] VkDescriptorSet get_descriptor_set() const { return descriptor_set_; }
	[[nodiscard]] int get_image_memory_handle() const;

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