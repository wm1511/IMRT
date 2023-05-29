#pragma once
#include "Frame.hpp"
#include "../abstract/IDrawable.hpp"
#include "../abstract/IRenderer.hpp"
#include "../info/RenderInfo.hpp"
#include "../info/WorldInfo.hpp"
#include "../info/SkyInfo.hpp"
#include "../info/CameraInfo.hpp"

class RtInterface final : public IDrawable
{
public:
	RtInterface();
	~RtInterface() override;

	RtInterface(const RtInterface&) = delete;
	RtInterface(RtInterface&&) = delete;
	RtInterface& operator=(const RtInterface&) = delete;
	RtInterface& operator=(RtInterface&&) = delete;

	void draw() override;

private:
	void check_cuda_optix();
	void move_camera();
	void edit_settings();
	void edit_camera();
	void add_texture();
	void edit_texture();
	void add_material();
	void edit_material();
	void add_object();
	void edit_object();
	void edit_sky();
	void save_image();

	RenderDevice render_device_{RenderDevice::CPU};
	std::unique_ptr<Frame> frame_ = nullptr;
	std::chrono::time_point<std::chrono::steady_clock> last_frame_time_ = std::chrono::high_resolution_clock::now();

	std::unique_ptr<IRenderer> renderer_ = nullptr;
	WorldInfo world_info_{};
	RenderInfo render_info_{};
	SkyInfo sky_info_{};
	CameraInfo camera_info_{};

	float camera_movement_speed_ = 0.002f, camera_rotation_speed_ = 0.002f;
	uint64_t render_time_ = 0, frames_rendered_ = 0;
	bool is_rendering_ = false, supports_cuda_ = false, supports_optix_ = false;
};