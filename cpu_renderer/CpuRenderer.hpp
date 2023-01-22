#pragma once
#include "../scene/RtInfo.hpp"
#include "../abstract/IRenderer.hpp"

class CpuRenderer final : public IRenderer
{
public:
	explicit CpuRenderer(const RenderInfo* render_info);
	void render(float* image_data, uint32_t width, uint32_t height) override;
	void recreate_camera(uint32_t width, uint32_t height) override;
	void recreate_image(uint32_t width, uint32_t height) override;
	void recreate_world(MaterialInfo** material_data, ObjectInfo** object_data) override;

private:
	const RenderInfo* render_info_;
};