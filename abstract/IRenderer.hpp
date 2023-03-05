#pragma once
#include <cstdint>

class CudaRenderer;

class IRenderer
{
public:
	virtual ~IRenderer() = default;

	virtual void render(float* image_data) = 0;
	virtual void refresh_buffer() = 0;
	virtual void refresh_camera() = 0;
	virtual void refresh_object(int32_t index) const = 0;
	virtual void refresh_material(int32_t index) const = 0;
	virtual void refresh_texture(int32_t index) const = 0;
	virtual void recreate_camera() = 0;
	virtual void recreate_image() = 0;
	virtual void recreate_sky() = 0;
	virtual void allocate_world() = 0;
	virtual void deallocate_world() const = 0;
};
