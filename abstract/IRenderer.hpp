#pragma once
#include <cstdint>

class IRenderer
{
public:
	virtual ~IRenderer() = default;
	virtual void render(float* image_data, uint32_t width, uint32_t height) = 0;
	// TODO Use functions declared below in renderer to separate work
	/*virtual void recreate_camera(void** camera_pointer, uint32_t width, uint32_t height) = 0;
	virtual void recreate_scene(void** objects_pointer, void** world_pointer) = 0;*/
};
