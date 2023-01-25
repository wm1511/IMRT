#include "CpuRenderer.hpp"
#include "../cuda_renderer/Math.cuh"

#include <cstdint>

CpuRenderer::CpuRenderer(const RenderInfo* render_info) : render_info_(render_info)
{
}

void CpuRenderer::render(float* image_data)
{
	const uint32_t width = render_info_->width;
    const uint32_t height = render_info_->height;
	uint32_t state = 435453453;

 #pragma omp parallel for schedule(dynamic)
	for (int32_t y = 0; y < static_cast<int32_t>(height); y++)
	{
		for (int32_t x = 0; x < static_cast<int32_t>(width); x++)
		{
			image_data[4 * (y * width + x)] = pcg_rxs_m_xs(&state);//static_cast<float>(rand()) / RAND_MAX;
			image_data[4 * (y * width + x) + 1] = pcg_rxs_m_xs(&state);//static_cast<float>(rand()) / RAND_MAX;
			image_data[4 * (y * width + x) + 2] = pcg_rxs_m_xs(&state);//static_cast<float>(rand()) / RAND_MAX;
			image_data[4 * (y * width + x) + 3] = 1.0f;
		}
	}
}

void CpuRenderer::recreate_camera()
{
}

void CpuRenderer::recreate_image()
{
}

void CpuRenderer::recreate_world()
{
}
