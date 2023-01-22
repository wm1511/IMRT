#include "CpuRenderer.hpp"

#include <random>

CpuRenderer::CpuRenderer(const RenderInfo* render_info) : render_info_(render_info)
{
}

void CpuRenderer::render(float* image_data, const uint32_t width, const uint32_t height)
{
 #pragma omp parallel for schedule(dynamic)
	for (int32_t y = 0; y < static_cast<int32_t>(height); y++)
	{
		for (int32_t x = 0; x < static_cast<int32_t>(width); x++)
		{
			image_data[4 * (y * width + x)] = static_cast<float>(rand()) / RAND_MAX;
			image_data[4 * (y * width + x) + 1] = static_cast<float>(rand()) / RAND_MAX;
			image_data[4 * (y * width + x) + 2] = static_cast<float>(rand()) / RAND_MAX;
			image_data[4 * (y * width + x) + 3] = 1.0f;
		}
	}
}

void CpuRenderer::recreate_camera(uint32_t width, uint32_t height)
{
}

void CpuRenderer::recreate_image(uint32_t width, uint32_t height)
{
}

void CpuRenderer::recreate_world(MaterialInfo** material_data, ObjectInfo** object_data)
{
}
