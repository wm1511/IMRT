#pragma once
#include "../abstract/IRenderer.hpp"
#include "../info/RenderInfo.hpp"
#include "../info/WorldInfo.hpp"
#include "../common/Color.hpp"

class CpuRenderer final : public IRenderer
{
public:
	CpuRenderer(const RenderInfo* render_info, WorldInfo* world_info, const SkyInfo* sky_info, const CameraInfo* camera_info);
	~CpuRenderer() override;

	CpuRenderer(const CpuRenderer&) = delete;
	CpuRenderer(CpuRenderer&&) = delete;
	CpuRenderer& operator=(const CpuRenderer&) = delete;
	CpuRenderer& operator=(CpuRenderer&&) = delete;

	void render() override;
	void refresh_buffer() override;
	void recreate_image() override;
	void recreate_sky() override;
	void allocate_world() override;
	void deallocate_world() override;

private:
	void random_init() const;
	void random_refresh() const;
	void render_static() const;
	void render_progressive() const;

	const RenderInfo* render_info_ = nullptr;
	WorldInfo* world_info_ = nullptr;
	const SkyInfo* sky_info_ = nullptr;
	const CameraInfo* camera_info_ = nullptr;

	uint4* xoshiro_state_ = nullptr, * xoshiro_initial_ = nullptr;
    float4* accumulation_buffer_ = nullptr;
    World* world_ = nullptr;
};