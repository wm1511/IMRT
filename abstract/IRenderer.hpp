#pragma once

class IRenderer
{
public:
	virtual ~IRenderer() = default;
	virtual void render(float* image_data) = 0;
	virtual void recreate_camera() = 0;
	virtual void recreate_image() = 0;
	virtual void recreate_world() = 0;
};
