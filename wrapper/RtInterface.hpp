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
	bool edit_camera();
	bool edit_scene();

	std::unique_ptr<Image> image_ = nullptr;
	std::unique_ptr<IRenderer> renderer_ = nullptr;
	RtInfo rt_info_{};
	float* image_data_ = nullptr;
	uint32_t height_ = 0, width_ = 0;

	uint64_t render_time_ = 0, frames_rendered_ = 0;
	bool is_rendering_ = false, camera_changed_ = false;
};