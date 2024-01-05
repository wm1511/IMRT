// Copyright Wiktor Merta 2023
#pragma once
#include "tiny_obj_loader.h"

#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_vulkan.h"

#include <filesystem>
#include <fstream>
#include <chrono>
#include <string>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
