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
	bool edit_camera();
	bool add_material();
	bool edit_material() const;
	bool add_object();
	bool edit_object();

	std::unique_ptr<Image> image_ = nullptr;
	std::unique_ptr<IRenderer> renderer_ = nullptr;
	RenderInfo render_info_{};
	
	ObjectInfo** object_data_ = nullptr;
	MaterialInfo** material_data_ = nullptr;
	float* image_data_ = nullptr;
	uint32_t height_ = 0, width_ = 0;

	uint64_t render_time_ = 0, frames_rendered_ = 0;
	bool is_rendering_{false}, camera_edited_{false}, object_edited_{false}, object_added_{false}, material_edited_{false}, material_added_{false};
};