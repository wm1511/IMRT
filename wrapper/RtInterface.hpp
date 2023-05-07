#pragma once
#include "Frame.hpp"
#include "../abstract/IDrawable.hpp"
#include "../abstract/IRenderer.hpp"
#include "../info/RenderInfo.hpp"
#include "../info/WorldInfo.hpp"
#include "../info/SkyInfo.hpp"

class RtInterface final : public IDrawable
{
public:
	void draw() override;

private:
	void move_camera() const;
	void edit_settings() const;
	void edit_camera();
	void add_texture() const;
	void edit_texture() const;
	void add_material() const;
	void edit_material() const;
	void add_object() const;
	void edit_object() const;
	void edit_sky() const;
	void save_image() const;

	std::unique_ptr<Frame> frame_ = nullptr;
	std::chrono::time_point<std::chrono::steady_clock> last_frame_time_ = std::chrono::high_resolution_clock::now();

	std::unique_ptr<IRenderer> renderer_ = nullptr;
	RenderInfo* render_info_ = nullptr;
	WorldInfo* world_info_ = nullptr;
	SkyInfo* sky_info_ = nullptr;

	int32_t render_device_{};
	float camera_movement_speed_ = 0.002f, camera_rotation_speed_ = 0.002f;
	uint64_t render_time_ = 0, frames_rendered_ = 0;
	bool is_rendering_ = false;
};