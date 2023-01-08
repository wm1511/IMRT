#pragma once
#include "../abstract/IRenderer.hpp"
#include "../wrapper/RtInfo.hpp"

class CudaRenderer final : public IRenderer
{
public:
	explicit CudaRenderer(const RtInfo& rt_info);
	void render(uint32_t* image_data, uint32_t width, uint32_t height) override;

private:
	RtInfo rt_info_;
};