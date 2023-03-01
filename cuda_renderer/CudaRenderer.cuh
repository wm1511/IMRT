#pragma once
#include "../abstract/IRenderer.hpp"
#include "../info/RenderInfo.hpp"
#include "../info/WorldInfo.hpp"

#include "../common/Camera.cuh"
#include "../common/Color.cuh"

class CudaRenderer final : public IRenderer
{
public:
	CudaRenderer(const RenderInfo* render_info, const WorldInfo* world_info, SkyInfo* sky_info);
	~CudaRenderer() override;

	CudaRenderer(const CudaRenderer&) = delete;
	CudaRenderer(CudaRenderer&&) = delete;
	CudaRenderer& operator=(const CudaRenderer&) = delete;
	CudaRenderer& operator=(CudaRenderer&&) = delete;

	void render(float* image_data) override;
	void refresh_buffer() override;
	void refresh_camera() override;
	void refresh_material(int32_t index) const override;
	void refresh_object(int32_t index) const override;
	void recreate_camera() override;
	void recreate_image() override;
	void recreate_sky() override;
	void allocate_world() override;
	void deallocate_world() const override;

private:
	const RenderInfo* render_info_;
	const WorldInfo* world_info_;
	SkyInfo* sky_info_;
	dim3 blocks_;
	dim3 threads_;

	uint4* xoshiro_state_ = nullptr;
    float4* frame_buffer_ = nullptr, * accumulation_buffer_ = nullptr;
    MaterialInfo** device_material_data_ = nullptr, ** host_material_data_ = nullptr;
    ObjectInfo** device_object_data_ = nullptr, ** host_object_data_ = nullptr;
    World** world_ = nullptr;
    Camera** camera_ = nullptr;
};
