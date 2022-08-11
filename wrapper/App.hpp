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

#include "IDrawable.hpp"

#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan/vulkan.h>

#include <string>
#include <memory>
#include <stdexcept>


struct AppInfo
{
	std::string name = "Default mWindow name";
	int32_t width = 1280;
	int32_t height = 720;
	float fontSize = 20.0f;
};

class App
{
public:
	explicit App(AppInfo& appInfo);
	~App();

	void run();

	template <typename T>
	void setInterface()
	{
		if (!std::is_base_of_v<IDrawable, T>)
			throw std::invalid_argument("Passed class is not a class derived of IDrawable");

		mAppInterface = std::make_unique<T>();
	}

	explicit App(const App&) = delete;
	explicit App(App&&) = delete;
	App operator= (const App&) = delete;
	App operator= (App&&) = delete;

	static VkInstance getInstance();
	static VkPhysicalDevice getPhysicalDevice();
	static VkDevice getDevice();

	static VkCommandBuffer getCommandBuffer();
	static void flushCommandBuffer(VkCommandBuffer commandBuffer);

private:
	void initialize();
	void terminate();

	AppInfo mInfo;
	GLFWwindow* mWindow;
	std::unique_ptr<IDrawable> mAppInterface;
};
