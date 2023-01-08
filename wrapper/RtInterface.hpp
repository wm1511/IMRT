#pragma once
#include "../abstract/IDrawable.hpp"
#include "../abstract/IRenderer.hpp"
#include "RtInfo.hpp"
#include "Image.hpp"

class RtInterface final : public IDrawable
{
public:
	void draw() override;

private:
	std::unique_ptr<Image> image_ = nullptr;
	std::unique_ptr<IRenderer> renderer_ = nullptr;
	uint32_t height_ = 0, width_ = 0;
	uint32_t* image_data_ = nullptr;
	uint64_t frames_rendered_ = 0, render_time_ = 0;
	RtInfo rt_info_;
	bool is_rendering_ = false;
};