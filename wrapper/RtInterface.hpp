#pragma once
#include "../abstract/IDrawable.hpp"
#include "../abstract/IRenderer.hpp"
#include "../info/RenderInfo.hpp"
#include "../info/WorldInfo.hpp"
#include "Image.hpp"

class RtInterface final : public IDrawable
{
public:
	RtInterface() = default;
	~RtInterface() override;

	RtInterface(const RtInterface&) = delete;
	RtInterface(RtInterface&&) = delete;
	RtInterface& operator=(const RtInterface&) = delete;
	RtInterface& operator=(RtInterface&&) = delete;

	void draw() override;

private:
	void move_camera();
	void edit_settings();
	void edit_camera();
	void add_material();
	void edit_material();
	void add_object();
	void edit_object();
	void edit_sky();
	void save_image() const;

	std::unique_ptr<Image> image_ = nullptr;
	std::unique_ptr<IRenderer> renderer_ = nullptr;
	RenderInfo render_info_{};
	WorldInfo world_info_{};

	float* image_data_ = nullptr;
	float camera_movement_speed_ = 0.08f, camera_rotation_speed_ = 0.002f;
	uint64_t render_time_ = 0, frames_rendered_ = 0;
	bool is_rendering_ = false;
};