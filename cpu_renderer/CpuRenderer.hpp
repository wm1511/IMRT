#pragma once
#include "../wrapper/RtInfo.hpp"
#include "../abstract/IRenderer.hpp"

class CpuRenderer final : public IRenderer
{
public:
	explicit CpuRenderer(const RtInfo* rt_info);
	void render(float* image_data, uint32_t width, uint32_t height) override;

private:
	const RtInfo* rt_info_;
};