#pragma once

#include "IDrawable.hpp"

#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan/vulkan.h>

#include <string>
#include <memory>
#include <stdexcept>

namespace imrt
{
	struct AppInfo
	{
		std::string name = "Default window name";
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
			{
				fprintf(stderr, "Passed class is not a class derived of IDrawable");
				abort();
			}
			appInterface = std::make_unique<T>();
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

		AppInfo info;
		GLFWwindow* window;
		std::unique_ptr<IDrawable> appInterface;
	};
}