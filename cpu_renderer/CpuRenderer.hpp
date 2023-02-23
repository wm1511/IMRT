#pragma once
#include "../abstract/IRenderer.hpp"
#include "../info/RenderInfo.hpp"
#include "../info/WorldInfo.hpp"

#include "../common/Color.cuh"
#include "../common/Camera.cuh"

class CpuRenderer final : public IRenderer
{
public:
	CpuRenderer(const RenderInfo* render_info, const WorldInfo* world_info);
	~CpuRenderer() override;

	CpuRenderer(const CpuRenderer&) = delete;
	CpuRenderer(CpuRenderer&&) = delete;
	CpuRenderer& operator=(const CpuRenderer&) = delete;
	CpuRenderer& operator=(CpuRenderer&&) = delete;

	void render(float* image_data) override;
	void refresh_buffer() override;
	void refresh_camera() override;
	void refresh_object(int32_t index) const override;
	void refresh_material(int32_t index) const override;
	void recreate_camera() override;
	void recreate_image() override;
	void recreate_sky() override;
	void allocate_world() override;
	void deallocate_world() const override;

private:
	const RenderInfo* render_info_;
	const WorldInfo* world_info_;

	uint4* xoshiro_state_ = nullptr;
    float4* accumulation_buffer_ = nullptr;
	float3* hdr_data_ = nullptr;
    MaterialInfo** material_data_ = nullptr;
    ObjectInfo** object_data_ = nullptr;
    World* world_ = nullptr;
    Camera* camera_ = nullptr;

	void random_init() const;
};