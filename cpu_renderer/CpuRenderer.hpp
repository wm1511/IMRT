#pragma once
#include "../scene/RtInfo.hpp"
#include "../abstract/IRenderer.hpp"

class CpuRenderer final : public IRenderer
{
public:
	explicit CpuRenderer(const RenderInfo* render_info);
	void render(float* image_data) override;
	void recreate_camera() override;
	void recreate_image() override;
	void recreate_world() override;

private:
	const RenderInfo* render_info_;
};