#pragma once
#include "LaunchParams.hpp"
#include "../common/Camera.cuh"
#include "../common/Color.cuh"

#include "../info/RenderInfo.hpp"
#include "../info/WorldInfo.hpp"
#include "../abstract/IRenderer.hpp"

#include <cuda.h>
#include <optix_types.h>

template <typename T>
struct SbtRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

class OptixRenderer final : public IRenderer
{
public:
	OptixRenderer(const RenderInfo* render_info, const WorldInfo* world_info, const SkyInfo* sky_info);
	~OptixRenderer() override;

	OptixRenderer(const OptixRenderer&) = delete;
	OptixRenderer(OptixRenderer&&) = delete;
	OptixRenderer& operator=(const OptixRenderer&) = delete;
	OptixRenderer& operator=(OptixRenderer&&) = delete;

	void render() override;
	void refresh_buffer() override;
	void refresh_camera() override;
	void refresh_object(int32_t index) const override;
	void refresh_material(int32_t index) const override;
	void refresh_texture(int32_t index) const override;
	void recreate_camera() override;
	void recreate_image() override;
	void recreate_sky() override;
	void allocate_world() override;
	void deallocate_world() const override;

private:
	void init_optix();
	void create_modules();
	void create_programs();
	void create_pipeline();
	void create_sbt();

	const RenderInfo* render_info_ = nullptr;
	const WorldInfo* world_info_ = nullptr;
	const SkyInfo* sky_info_ = nullptr;

	OptixDeviceContext context_ = nullptr;
	OptixModule module_ = nullptr;
	OptixPipeline pipeline_ = nullptr;
	OptixShaderBindingTable sbt_{};

	std::vector<OptixProgramGroup> raygen_programs_{};
	std::vector<OptixProgramGroup> miss_programs_{};
	std::vector<OptixProgramGroup> hit_programs_{};
	CUdeviceptr raygen_record_buffer_{};
	CUdeviceptr miss_record_buffer_{};
	CUdeviceptr hit_record_buffer_{};

	LaunchParams launch_params_{};
	CUdeviceptr device_launch_params_{};
	CUdeviceptr frame_buffer_{};
	CUstream stream_{};
};