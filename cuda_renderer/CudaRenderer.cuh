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
	CudaRenderer(const RenderInfo* render_info, MaterialInfo** material_data, ObjectInfo** object_data,	uint32_t width, uint32_t height);
	~CudaRenderer() override;

	CudaRenderer(const CudaRenderer&) = delete;
	CudaRenderer(CudaRenderer&&) = delete;
	CudaRenderer operator=(const CudaRenderer&) = delete;
	CudaRenderer operator=(CudaRenderer&&) = delete;

	void render(float* image_data, uint32_t width, uint32_t height) override;
	void recreate_camera(uint32_t width, uint32_t height) override;
	void recreate_image(uint32_t width, uint32_t height) override;
	void recreate_world(MaterialInfo** material_data, ObjectInfo** object_data) override;

private:
	const RenderInfo* render_info_;
	const int32_t thread_x_ = 16;
	const int32_t thread_y_ = 16;

	MaterialInfo** device_material_data_ = nullptr, ** host_material_data_ = nullptr;
	ObjectInfo** device_object_data_ = nullptr, ** host_object_data_ = nullptr;
    float4* frame_buffer_ = nullptr;
    uint32_t* random_state_ = nullptr;
    Material** materials_list_ = nullptr;
    Primitive** primitives_list_ = nullptr;
    World** world_ = nullptr;
    Camera** camera_ = nullptr;
};
