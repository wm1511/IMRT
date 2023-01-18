#pragma once
#include "../abstract/IDrawable.hpp"
#include "../abstract/IRenderer.hpp"
#include "../scene/RtInfo.hpp"
#include "../scene/ObjectInfo.hpp"
#include "Image.hpp"

#include <vector>

class RtInterface final : public IDrawable
{
public:
	void init() override;
	void draw() override;

private:
	bool edit_camera();
	bool edit_scene();

	std::unique_ptr<Image> image_ = nullptr;
	std::unique_ptr<IRenderer> renderer_ = nullptr;
	std::vector<std::shared_ptr<ObjectInfo>> object_data_{};
	std::vector<std::shared_ptr<MaterialInfo>> material_data_{};

	RenderInfo render_info_{};
	float* image_data_ = nullptr;
	uint32_t height_ = 0, width_ = 0;

	uint64_t render_time_ = 0, frames_rendered_ = 0;
	bool is_rendering_ = false, camera_changed_ = false, scene_changed_ = false;
};