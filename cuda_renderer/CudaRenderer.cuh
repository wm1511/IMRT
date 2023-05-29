#pragma once
#include "../common/Color.hpp"

#include "../info/RenderInfo.hpp"
#include "../info/WorldInfo.hpp"
#include "../abstract/IRenderer.hpp"

class CudaRenderer final : public IRenderer
{
public:
	CudaRenderer(const RenderInfo* render_info, WorldInfo* world_info, const SkyInfo* sky_info, const CameraInfo* camera_info);
	~CudaRenderer() override;

	CudaRenderer(const CudaRenderer&) = delete;
	CudaRenderer(CudaRenderer&&) = delete;
	CudaRenderer& operator=(const CudaRenderer&) = delete;
	CudaRenderer& operator=(CudaRenderer&&) = delete;

	void render() override;
	void refresh_buffer() override;
	void refresh_texture(int32_t index) const override;
	void refresh_material(int32_t index) const override;
	void refresh_object(int32_t index) override;
	void recreate_image() override;
	void recreate_sky() override;
	void map_frame_memory() override;
	void allocate_world() override;
	void deallocate_world() override;

private:
	const RenderInfo* render_info_ = nullptr;
	WorldInfo* world_info_ = nullptr;
	const SkyInfo* sky_info_ = nullptr;
	const CameraInfo* camera_info_ = nullptr;

	dim3 blocks_;
	dim3 threads_;

	uint4* xoshiro_state_ = nullptr, * xoshiro_initial_ = nullptr;
	float4* accumulation_buffer_ = nullptr;
	Texture* d_texture_data_ = nullptr;
	Material* d_material_data_ = nullptr;
	Object* d_object_data_ = nullptr;
	World* world_ = nullptr;
};
