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

	void setData(const void* data);

	[[nodiscard]] VkDescriptorSet getDescriptorSet() const { return mDescriptorSet; }

	[[nodiscard]] uint32_t getWidth() const { return mWidth; }
	[[nodiscard]] uint32_t getHeight() const { return mHeight; }

private:
	void allocateMemory();
	void releaseMemory();

	uint32_t mWidth = 0, mHeight = 0;

	VkImage mImage = nullptr;
	VkImageView mImageView = nullptr;
	VkDeviceMemory mMemory = nullptr;
	VkSampler mSampler = nullptr;
	VkBuffer mStagingBuffer = nullptr;
	VkDeviceMemory mStagingBufferMemory = nullptr;

	size_t mRequiredMemorySize = 0;

	VkDescriptorSet mDescriptorSet = nullptr;
};