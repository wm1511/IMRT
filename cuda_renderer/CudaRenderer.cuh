#pragma once
#include "../abstract/IRenderer.hpp"
#include "../scene/RtInfo.hpp"
#include "../scene/ObjectInfo.hpp"
#include "../scene/MaterialInfo.hpp"

#include "World.cuh"
#include "Camera.cuh"

class CudaRenderer final : public IRenderer
{
public:
	explicit CudaRenderer(const RenderInfo* render_info);
	~CudaRenderer() override;

	CudaRenderer(const CudaRenderer&) = delete;
	CudaRenderer(CudaRenderer&&) = delete;
	CudaRenderer operator=(const CudaRenderer&) = delete;
	CudaRenderer operator=(CudaRenderer&&) = delete;

	void render(float* image_data) override;
	void recreate_camera() override;
	void recreate_image() override;
	void recreate_world() override;

private:
	void allocate_world();
	void deallocate_world() const;

	const RenderInfo* render_info_;

	MaterialInfo** device_material_data_ = nullptr, ** host_material_data_ = nullptr;
	ObjectInfo** device_object_data_ = nullptr, ** host_object_data_ = nullptr;
    float4* frame_buffer_ = nullptr, * accumulation_buffer_ = nullptr;
    uint32_t* random_state_ = nullptr;
    Material** materials_list_ = nullptr;
    Primitive** primitives_list_ = nullptr;
    World** world_ = nullptr;
    Camera** camera_ = nullptr;
};
