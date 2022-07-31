#include "Image.hpp"

#include "../imgui/imgui_impl_vulkan.h"

namespace imrt
{

	static uint32_t getDeviceMemoryType(VkMemoryPropertyFlags properties, uint32_t typeBits)
	{
		VkPhysicalDeviceMemoryProperties deviceMemoryProperties;
		vkGetPhysicalDeviceMemoryProperties(App::getPhysicalDevice(), &deviceMemoryProperties);
		for (uint32_t i = 0; i < deviceMemoryProperties.memoryTypeCount; i++)
		{
			if ((deviceMemoryProperties.memoryTypes[i].propertyFlags & properties) == properties && typeBits & (1 << i))
				return i;
		}
		
		return UINT32_MAX;
	}

	Image::Image(const uint32_t width, const uint32_t height, const void* data): width(width), height(height)
	{
		allocateMemory();
		if (data)
			setData(data);
	}

	Image::~Image()
	{
		releaseMemory();
	}

	void Image::allocateMemory()
	{
		VkDevice device = App::getDevice();

		VkImageCreateInfo imageInfo = {};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS)
			throw std::runtime_error("Failed to create image");

		VkMemoryRequirements requirements;
		vkGetImageMemoryRequirements(device, image, &requirements);
		VkMemoryAllocateInfo allocateInfo = {};
		allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocateInfo.allocationSize = requirements.size;
		allocateInfo.memoryTypeIndex = getDeviceMemoryType(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, requirements.memoryTypeBits);
		if (vkAllocateMemory(device, &allocateInfo, nullptr, &memory) != VK_SUCCESS)
			throw std::runtime_error("Failed to allocate image memory");
		if (vkBindImageMemory(device, image, memory, 0 != VK_SUCCESS))
			throw std::runtime_error("Failed to bind memory to image");

		VkImageViewCreateInfo imageViewInfo = {};
		imageViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		imageViewInfo.image = image;
		imageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		imageViewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
		imageViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageViewInfo.subresourceRange.levelCount = 1;
		imageViewInfo.subresourceRange.layerCount = 1;
		if (vkCreateImageView(device, &imageViewInfo, nullptr, &imageView) != VK_SUCCESS)
			throw std::runtime_error("Failed to create image view");

		VkSamplerCreateInfo samplerInfo = {};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.minLod = -1000;
		samplerInfo.maxLod = 1000;
		samplerInfo.maxAnisotropy = 1.0f;
		if (vkCreateSampler(device, &samplerInfo, nullptr, &sampler) != VK_SUCCESS)
			throw std::runtime_error("Failed to create sampler");

		descriptorSet = ImGui_ImplVulkan_AddTexture(sampler, imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	}

	void Image::releaseMemory()
	{
		VkDevice device = App::getDevice();
		vkDeviceWaitIdle(device);

		vkDestroySampler(device, sampler, nullptr);
		vkDestroyImageView(device, imageView, nullptr);
		vkDestroyImage(device, image, nullptr);
		vkFreeMemory(device, memory, nullptr);
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);

		sampler = nullptr;
		imageView = nullptr;
		image = nullptr;
		memory = nullptr;
		stagingBuffer = nullptr;
		stagingBufferMemory = nullptr;
	}

	void Image::setData(const void* data)
	{
		VkDevice device = App::getDevice();
		const uint64_t stagingBufferSize = static_cast<uint64_t>(4) * width * height;

		if (!stagingBuffer)
		{
			VkBufferCreateInfo bufferInfo = {};
			bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			bufferInfo.size = stagingBufferSize;
			bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
			bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			if (vkCreateBuffer(device, &bufferInfo, nullptr, &stagingBuffer) != VK_SUCCESS)
				throw std::runtime_error("Failed to create staging buffer");
			VkMemoryRequirements requirements;
			vkGetBufferMemoryRequirements(device, stagingBuffer, &requirements);
			requiredMemorySize = requirements.size;
			VkMemoryAllocateInfo allocateInfo = {};
			allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocateInfo.allocationSize = requirements.size;
			allocateInfo.memoryTypeIndex = getDeviceMemoryType(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, requirements.memoryTypeBits);
			if (vkAllocateMemory(device, &allocateInfo, nullptr, &stagingBufferMemory) != VK_SUCCESS)
				throw std::runtime_error("Failed to allocate staging buffer memory");
			if (vkBindBufferMemory(device, stagingBuffer, stagingBufferMemory, 0) != VK_SUCCESS)
				throw std::runtime_error("Failed to bind memory to staging buffer");
		}

		char* map = nullptr;
		if (vkMapMemory(device, stagingBufferMemory, 0, requiredMemorySize, 0, reinterpret_cast<void**>(&map)) != VK_SUCCESS)
			throw std::runtime_error("Failed to map staging buffer memory to be GPU-readable");
		memcpy(map, data, stagingBufferSize);
		VkMappedMemoryRange range[1] = {};
		range[0].sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
		range[0].memory = stagingBufferMemory;
		range[0].size = requiredMemorySize;
		if (vkFlushMappedMemoryRanges(device, 1, range) != VK_SUCCESS)
			throw std::runtime_error("Failed to flush mapped memory to GPU");
		vkUnmapMemory(device, stagingBufferMemory);

		VkCommandBuffer commandBuffer = App::getCommandBuffer();

		VkImageMemoryBarrier copyBarrier = {};
		copyBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		copyBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		copyBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		copyBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		copyBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		copyBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		copyBarrier.image = image;
		copyBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		copyBarrier.subresourceRange.levelCount = 1;
		copyBarrier.subresourceRange.layerCount = 1;
		vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
		                     0, nullptr, 1, &copyBarrier);

		VkBufferImageCopy region = {};
		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.layerCount = 1;
		region.imageExtent.width = width;
		region.imageExtent.height = height;
		region.imageExtent.depth = 1;
		vkCmdCopyBufferToImage(commandBuffer, stagingBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		VkImageMemoryBarrier usageBarrier = {};
		usageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		usageBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		usageBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		usageBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		usageBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		usageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		usageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		usageBarrier.image = image;
		usageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		usageBarrier.subresourceRange.levelCount = 1;
		usageBarrier.subresourceRange.layerCount = 1;
		vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		                     0, 0, nullptr, 0, nullptr, 1, &usageBarrier);

		App::flushCommandBuffer(commandBuffer);
	}
}