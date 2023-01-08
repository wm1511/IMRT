#pragma once
#include <cstdint>

class IRenderer
{
public:
	virtual ~IRenderer() = default;
	virtual void render(uint32_t* image_data, uint32_t width, uint32_t height) = 0;
};
