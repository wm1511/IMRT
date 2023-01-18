#pragma once
#include "../abstract/IRenderer.hpp"
#include "../scene/RtInfo.hpp"

#include "World.cuh"
#include "Camera.cuh"

#include <device_launch_parameters.h>

class CudaRenderer final : public IRenderer
{
public:
	CudaRenderer(const RenderInfo* render_info, uint32_t width, uint32_t height);
	~CudaRenderer() override;

	CudaRenderer(const CudaRenderer&) = delete;
	CudaRenderer(CudaRenderer&&) = delete;
	CudaRenderer operator=(const CudaRenderer&) = delete;
	CudaRenderer operator=(CudaRenderer&&) = delete;

	void render(float* image_data, uint32_t width, uint32_t height) override;
	void recreate_camera(uint32_t width, uint32_t height) override;
	void recreate_image(uint32_t width, uint32_t height) override;
	void recreate_scene() override;

private:
	const RenderInfo* render_info_;
	const int32_t thread_x_ = 16;
	const int32_t thread_y_ = 16;

    float4* frame_buffer_ = nullptr;
    curandState* random_state_ = nullptr;
    Primitive** primitives_list_ = nullptr;
    World** world_ = nullptr;
    Camera** camera_ = nullptr;
};