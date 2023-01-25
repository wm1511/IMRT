#pragma once
#include "../abstract/IDrawable.hpp"
#include "../abstract/IRenderer.hpp"
#include "../scene/RtInfo.hpp"
#include "Image.hpp"

class RtInterface final : public IDrawable
{
public:
	RtInterface();
	~RtInterface() override;
	void draw() override;

private:
	void edit_settings();
	void edit_camera();
	void add_material();
	void edit_material();
	void add_object();
	void edit_object();

	std::unique_ptr<Image> image_ = nullptr;
	std::unique_ptr<IRenderer> renderer_ = nullptr;
	RenderInfo render_info_{};

	float* image_data_ = nullptr;
	uint64_t render_time_ = 0, frames_rendered_ = 0;
	bool is_rendering_ = false;
};