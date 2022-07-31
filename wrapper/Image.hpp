#pragma once

#include "App.hpp"

namespace imrt
{
	class Image
	{
	public:
		Image(uint32_t width, uint32_t height, const void* data = nullptr);
		Image(const Image&) = delete;
		Image(Image&&) = delete;
		Image operator= (const Image&) = delete;
		Image operator= (Image&&) = delete;
		~Image();

		void setData(const void* data);

		[[nodiscard]] VkDescriptorSet getDescriptorSet() const { return descriptorSet; }

		[[nodiscard]] uint32_t getWidth() const { return width; }
		[[nodiscard]] uint32_t getHeight() const { return height; }

	private:
		void allocateMemory();
		void releaseMemory();

		uint32_t width = 0, height = 0;

		VkImage image = nullptr;
		VkImageView imageView = nullptr;
		VkDeviceMemory memory = nullptr;
		VkSampler sampler = nullptr;
		VkBuffer stagingBuffer = nullptr;
		VkDeviceMemory stagingBufferMemory = nullptr;

		size_t requiredMemorySize = 0;

		VkDescriptorSet descriptorSet = nullptr;
	};
}