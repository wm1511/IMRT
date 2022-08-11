// Copyright (c) 2022, Wiktor Merta
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "Image.hpp"

#include "../imgui/imgui_impl_vulkan.h"

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

Image::Image(const uint32_t width, const uint32_t height, const void* data): mWidth(width), mHeight(height)
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
	constexpr VkFormat imageFormat = VK_FORMAT_R8G8B8A8_UNORM;

	VkImageCreateInfo imageInfo = {};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.format = imageFormat;
	imageInfo.extent.width = mWidth;
	imageInfo.extent.height = mHeight;
	imageInfo.extent.depth = 1;
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 1;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	if (vkCreateImage(device, &imageInfo, nullptr, &mImage) != VK_SUCCESS)
		throw std::runtime_error("Failed to create image");

	VkMemoryRequirements requirements;
	vkGetImageMemoryRequirements(device, mImage, &requirements);
	VkMemoryAllocateInfo allocateInfo = {};
	allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocateInfo.allocationSize = requirements.size;
	allocateInfo.memoryTypeIndex = getDeviceMemoryType(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, requirements.memoryTypeBits);
	if (vkAllocateMemory(device, &allocateInfo, nullptr, &mMemory) != VK_SUCCESS)
		throw std::runtime_error("Failed to allocate image memory");
	if (vkBindImageMemory(device, mImage, mMemory, 0 != VK_SUCCESS))
		throw std::runtime_error("Failed to bind memory to image");

	VkImageViewCreateInfo imageViewInfo = {};
	imageViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	imageViewInfo.image = mImage;
	imageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	imageViewInfo.format = imageFormat;
	imageViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	imageViewInfo.subresourceRange.levelCount = 1;
	imageViewInfo.subresourceRange.layerCount = 1;
	if (vkCreateImageView(device, &imageViewInfo, nullptr, &mImageView) != VK_SUCCESS)
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
	if (vkCreateSampler(device, &samplerInfo, nullptr, &mSampler) != VK_SUCCESS)
		throw std::runtime_error("Failed to create sampler");

	mDescriptorSet = ImGui_ImplVulkan_AddTexture(mSampler, mImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void Image::releaseMemory()
{
	VkDevice device = App::getDevice();
	vkDeviceWaitIdle(device);

	vkDestroySampler(device, mSampler, nullptr);
	vkDestroyImageView(device, mImageView, nullptr);
	vkDestroyImage(device, mImage, nullptr);
	vkFreeMemory(device, mMemory, nullptr);
	vkDestroyBuffer(device, mStagingBuffer, nullptr);
	vkFreeMemory(device, mStagingBufferMemory, nullptr);

	mSampler = nullptr;
	mImageView = nullptr;
	mImage = nullptr;
	mMemory = nullptr;
	mStagingBuffer = nullptr;
	mStagingBufferMemory = nullptr;
}

void Image::setData(const void* data)
{
	VkDevice device = App::getDevice();
	const uint64_t stagingBufferSize = static_cast<uint64_t>(4) * mWidth * mHeight;

	if (!mStagingBuffer)
	{
		VkBufferCreateInfo bufferInfo = {};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = stagingBufferSize;
		bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		if (vkCreateBuffer(device, &bufferInfo, nullptr, &mStagingBuffer) != VK_SUCCESS)
			throw std::runtime_error("Failed to create staging buffer");
		VkMemoryRequirements requirements;
		vkGetBufferMemoryRequirements(device, mStagingBuffer, &requirements);
		mRequiredMemorySize = requirements.size;
		VkMemoryAllocateInfo allocateInfo = {};
		allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocateInfo.allocationSize = requirements.size;
		allocateInfo.memoryTypeIndex = getDeviceMemoryType(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, requirements.memoryTypeBits);
		if (vkAllocateMemory(device, &allocateInfo, nullptr, &mStagingBufferMemory) != VK_SUCCESS)
			throw std::runtime_error("Failed to allocate staging buffer memory");
		if (vkBindBufferMemory(device, mStagingBuffer, mStagingBufferMemory, 0) != VK_SUCCESS)
			throw std::runtime_error("Failed to bind memory to staging buffer");
	}

	char* map = nullptr;
	if (vkMapMemory(device, mStagingBufferMemory, 0, mRequiredMemorySize, 0, reinterpret_cast<void**>(&map)) != VK_SUCCESS)
		throw std::runtime_error("Failed to map staging buffer memory to be GPU-readable");
	memcpy(map, data, stagingBufferSize);
	VkMappedMemoryRange range[1] = {};
	range[0].sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
	range[0].memory = mStagingBufferMemory;
	range[0].size = mRequiredMemorySize;
	if (vkFlushMappedMemoryRanges(device, 1, range) != VK_SUCCESS)
		throw std::runtime_error("Failed to flush mapped memory to GPU");
	vkUnmapMemory(device, mStagingBufferMemory);

	VkCommandBuffer commandBuffer = App::getCommandBuffer();

	VkImageMemoryBarrier copyBarrier = {};
	copyBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	copyBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	copyBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	copyBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	copyBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	copyBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	copyBarrier.image = mImage;
	copyBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	copyBarrier.subresourceRange.levelCount = 1;
	copyBarrier.subresourceRange.layerCount = 1;
	vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
	                     0, nullptr, 1, &copyBarrier);

	VkBufferImageCopy region = {};
	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.layerCount = 1;
	region.imageExtent.width = mWidth;
	region.imageExtent.height = mHeight;
	region.imageExtent.depth = 1;
	vkCmdCopyBufferToImage(commandBuffer, mStagingBuffer, mImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	VkImageMemoryBarrier usageBarrier = {};
	usageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	usageBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	usageBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	usageBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	usageBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	usageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	usageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	usageBarrier.image = mImage;
	usageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	usageBarrier.subresourceRange.levelCount = 1;
	usageBarrier.subresourceRange.layerCount = 1;
	vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
	                     0, 0, nullptr, 0, nullptr, 1, &usageBarrier);

	App::flushCommandBuffer(commandBuffer);
}
