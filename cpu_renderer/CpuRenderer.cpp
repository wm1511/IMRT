#include "CpuRenderer.hpp"

#include <random>

CpuRenderer::CpuRenderer(const RtInfo& rt_info) : rt_info_(rt_info)
{
}

void CpuRenderer::render(uint32_t* image_data, const uint32_t width, const uint32_t height)
{
 #pragma omp parallel for schedule(dynamic)
	for (int32_t y = 0; y < static_cast<int32_t>(height); y++)
	{
		for (int32_t x = 0; x < static_cast<int32_t>(width); x++)
		{
			image_data[y * width + x] = 0xff000000 | rand() / 128 << 16 | rand() / 128 << 8 | rand() / 128;
		}
	}
}
