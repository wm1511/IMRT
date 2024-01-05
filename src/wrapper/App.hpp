// Copyright Wiktor Merta 2023
#pragma once
#include "../abstract/IDrawable.hpp"

struct AppInfo
{
	std::string name = "Default window name";
	int32_t width = 1280;
	int32_t height = 720;
	float font_size = 20.0f;
};

class App
{
public:
	explicit App(AppInfo& app_info);
	~App();

	void run() const;

	// Choosing interface to be drawn
	template <typename T>
	void set_interface()
	{
		if (!std::is_base_of_v<IDrawable, T>)
			throw std::invalid_argument("Passed class is not a class derived of IDrawable");

		app_interface_ = std::make_unique<T>();
	}

	explicit App(const App&) = delete;
	explicit App(App&&) = delete;
	App& operator= (const App&) = delete;
	App& operator= (App&&) = delete;

	static VkInstance get_instance();
	static VkPhysicalDevice get_physical_device();
	static VkDevice get_device();
	static VkCommandBuffer get_command_buffer();
	static void flush_command_buffer(VkCommandBuffer command_buffer);

private:
	void initialize();
	void terminate();

	AppInfo info_{};
	GLFWwindow* window_ = nullptr;
	std::unique_ptr<IDrawable> app_interface_ = nullptr;
};
