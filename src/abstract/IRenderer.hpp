#pragma once
#include "../info/RenderInfo.hpp"
#include "../info/WorldInfo.hpp"
#include "../info/SkyInfo.hpp"
#include "../info/CameraInfo.hpp"

#include <cstdint>

class IRenderer
{
public:
	virtual ~IRenderer() = default;

	virtual void render() = 0;
	virtual void refresh_buffer() = 0;
	virtual void refresh_camera() {}
	virtual void refresh_object(int32_t) {}
	virtual void refresh_material(int32_t) const {}
	virtual void refresh_texture(int32_t) const {}
	virtual void recreate_image() = 0;
	virtual void recreate_sky() = 0;
	virtual void map_frame_memory() {}
	virtual void allocate_world() = 0;
	virtual void deallocate_world() = 0;

protected:
	const RenderInfo* render_info_ = nullptr;
	const WorldInfo* world_info_ = nullptr;
	const SkyInfo* sky_info_ = nullptr;
	const CameraInfo* camera_info_ = nullptr;
};
