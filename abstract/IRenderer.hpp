#pragma once
#include "../scene/ObjectInfo.hpp"
#include "../scene/MaterialInfo.hpp"

#include <cstdint>

class IRenderer
{
public:
	virtual ~IRenderer() = default;
	virtual void render(float* image_data, uint32_t width, uint32_t height) = 0;
	// TODO Use functions declared below in renderer to separate work
	virtual void recreate_camera(uint32_t width, uint32_t height) = 0;
	virtual void recreate_image(uint32_t width, uint32_t height) = 0;
	virtual void recreate_world(MaterialInfo** material_data, ObjectInfo** object_data) = 0;
};
