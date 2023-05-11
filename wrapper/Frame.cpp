#include "stdafx.h"
#include "Frame.hpp"

static uint32_t GetDeviceMemoryType(const VkMemoryPropertyFlags properties, const uint32_t type_bits)
{
	VkPhysicalDeviceMemoryProperties device_memory_properties;
	vkGetPhysicalDeviceMemoryProperties(App::get_physical_device(), &device_memory_properties);
	for (uint32_t i = 0; i < device_memory_properties.memoryTypeCount; i++)
	{
		if ((device_memory_properties.memoryTypes[i].propertyFlags & properties) == properties && type_bits & 1 << i)
			return i;
	}
	
	return ~0U;
}

Frame::Frame(const uint32_t width, const uint32_t height): width_(width), height_(height)
{
	allocate_memory();
}

Frame::~Frame()
{
	release_memory();
}

void Frame::allocate_memory()
{
	const VkDevice device = App::get_device();

	VkImageCreateInfo image_info = {};
	image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	image_info.imageType = VK_IMAGE_TYPE_2D;
	image_info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
	image_info.extent.width = width_;
	image_info.extent.height = height_;
	image_info.extent.depth = 1;
	image_info.mipLevels = 1;
	image_info.arrayLayers = 1;
	image_info.samples = VK_SAMPLE_COUNT_1_BIT;
	image_info.tiling = VK_IMAGE_TILING_LINEAR;
	image_info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	image_info.initialLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	VkExternalMemoryImageCreateInfoKHR external_memory_image_info = {};
	external_memory_image_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO_KHR;

#if defined(_WIN32)
            external_memory_image_info.handleTypes |= VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
#elif defined(__linux__) || defined(__APPLE__)
            external_memory_image_info.handleTypes |= VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif

	image_info.pNext = &external_memory_image_info;

	if (vkCreateImage(device, &image_info, nullptr, &image_) != VK_SUCCESS)
		throw std::runtime_error("Failed to create image");

	VkMemoryRequirements requirements;
	vkGetImageMemoryRequirements(device, image_, &requirements);
	required_memory_size_ = requirements.size;

	VkMemoryAllocateInfo allocate_info = {};
	allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocate_info.allocationSize = requirements.size;
	allocate_info.memoryTypeIndex = GetDeviceMemoryType(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, requirements.memoryTypeBits);

    VkExportMemoryAllocateInfoKHR export_memory_info = {};
    export_memory_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;

#if defined(_WIN32)
            export_memory_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
#elif defined(__linux__) || defined(__APPLE__)
            export_memory_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif

	allocate_info.pNext = &export_memory_info;

	if (vkAllocateMemory(device, &allocate_info, nullptr, &memory_) != VK_SUCCESS)
		throw std::runtime_error("Failed to allocate image memory");

	if (vkBindImageMemory(device, image_, memory_, 0) != VK_SUCCESS)
		throw std::runtime_error("Failed to bind memory to image");

	VkImageViewCreateInfo image_view_info = {};
	image_view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	image_view_info.image = image_;
	image_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
	image_view_info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
	image_view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	image_view_info.subresourceRange.levelCount = 1;
	image_view_info.subresourceRange.layerCount = 1;
	if (vkCreateImageView(device, &image_view_info, nullptr, &image_view_) != VK_SUCCESS)
		throw std::runtime_error("Failed to create image view");

	VkSamplerCreateInfo sampler_info = {};
	sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	sampler_info.magFilter = VK_FILTER_LINEAR;
	sampler_info.minFilter = VK_FILTER_LINEAR;
	sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	sampler_info.minLod = -1000;
	sampler_info.maxLod = 1000;
	sampler_info.maxAnisotropy = 1.0f;
	if (vkCreateSampler(device, &sampler_info, nullptr, &sampler_) != VK_SUCCESS)
		throw std::runtime_error("Failed to create sampler");

	descriptor_set_ = ImGui_ImplVulkan_AddTexture(sampler_, image_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void Frame::release_memory()
{
	const VkDevice device = App::get_device();
	vkDeviceWaitIdle(device);

	vkDestroySampler(device, sampler_, nullptr);
	vkDestroyImageView(device, image_view_, nullptr);
	vkDestroyImage(device, image_, nullptr);
	vkFreeMemory(device, memory_, nullptr);
	vkDestroyBuffer(device, staging_buffer_, nullptr);
	vkFreeMemory(device, staging_buffer_memory_, nullptr);

	sampler_ = nullptr;
	image_view_ = nullptr;
	image_ = nullptr;
	memory_ = nullptr;
	staging_buffer_ = nullptr;
	staging_buffer_memory_ = nullptr;
}

void Frame::set_data(const void* data)
{
	const VkDevice device = App::get_device();
	const uint64_t staging_buffer_size = 4 * sizeof(float) * width_ * height_;

	if (!staging_buffer_)
	{
		VkBufferCreateInfo buffer_info = {};
		buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		buffer_info.size = staging_buffer_size;
		buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateBuffer(device, &buffer_info, nullptr, &staging_buffer_) != VK_SUCCESS)
			throw std::runtime_error("Failed to create staging buffer");

		VkMemoryRequirements requirements;
		vkGetBufferMemoryRequirements(device, staging_buffer_, &requirements);
		required_memory_size_ = requirements.size;

		VkMemoryAllocateInfo allocate_info = {};
		allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocate_info.allocationSize = requirements.size;
		allocate_info.memoryTypeIndex = GetDeviceMemoryType(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, requirements.memoryTypeBits);

		if (vkAllocateMemory(device, &allocate_info, nullptr, &staging_buffer_memory_) != VK_SUCCESS)
			throw std::runtime_error("Failed to allocate staging buffer memory");

		if (vkBindBufferMemory(device, staging_buffer_, staging_buffer_memory_, 0) != VK_SUCCESS)
			throw std::runtime_error("Failed to bind memory to staging buffer");
	}

	float* map = nullptr;
	if (vkMapMemory(device, staging_buffer_memory_, 0, required_memory_size_, 0, reinterpret_cast<void**>(&map)) != VK_SUCCESS)
		throw std::runtime_error("Failed to map staging buffer memory to be GPU-readable");
	memcpy(map, data, staging_buffer_size);

	VkMappedMemoryRange range[1] = {};
	range[0].sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
	range[0].memory = staging_buffer_memory_;
	range[0].size = required_memory_size_;

	if (vkFlushMappedMemoryRanges(device, 1, range) != VK_SUCCESS)
		throw std::runtime_error("Failed to flush mapped memory to GPU");

	vkUnmapMemory(device, staging_buffer_memory_);

	const VkCommandBuffer command_buffer = App::get_command_buffer();

	VkImageMemoryBarrier copy_barrier = {};
	copy_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	copy_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	copy_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	copy_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	copy_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	copy_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	copy_barrier.image = image_;
	copy_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	copy_barrier.subresourceRange.levelCount = 1;
	copy_barrier.subresourceRange.layerCount = 1;
	vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
	                     0, nullptr, 1, &copy_barrier);

	VkBufferImageCopy region = {};
	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.layerCount = 1;
	region.imageExtent.width = width_;
	region.imageExtent.height = height_;
	region.imageExtent.depth = 1;
	vkCmdCopyBufferToImage(command_buffer, staging_buffer_, image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	VkImageMemoryBarrier usage_barrier = {};
	usage_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	usage_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	usage_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	usage_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	usage_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	usage_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	usage_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	usage_barrier.image = image_;
	usage_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	usage_barrier.subresourceRange.levelCount = 1;
	usage_barrier.subresourceRange.layerCount = 1;
	vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
	                     0, 0, nullptr, 0, nullptr, 1, &usage_barrier);

	App::flush_command_buffer(command_buffer);
}

void* Frame::get_image_memory_handle() const
{
	const VkDevice device = App::get_device();

#if defined(_WIN32)
	void* handle;

	VkMemoryGetWin32HandleInfoKHR handle_info = {};
	handle_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
	handle_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
	handle_info.memory = memory_;

	const auto vkGetMemoryWin32HandleKHR = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR"));

    if (!vkGetMemoryWin32HandleKHR)
        throw std::runtime_error("Failed to obtain vkGetMemoryWin32HandleKHR function pointer");

    if (vkGetMemoryWin32HandleKHR(device, &handle_info, &handle) != VK_SUCCESS)
		throw std::runtime_error("Failed to execute vkGetMemoryWin32HandleKHR");

	return handle;

#elif defined(__linux__) || defined(__APPLE__)
	int handle = -1;

	VkMemoryGetFdInfoKHR handle_info = {};
	handle_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
	handle_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
	handle_info.memory = memory_;

	const auto vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR) vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");

    if (!vkGetMemoryFdKHR)
        throw std::runtime_error("Failed to obtain vkGetMemoryFdKHR function pointer");

    if (vkGetMemoryFdKHR(device, &handle_info, &handle) != VK_SUCCESS)
		throw std::runtime_error("Failed to execute vkGetMemoryFdKHR");

	return reinterpret_cast<void*>(handle);
#endif
}
