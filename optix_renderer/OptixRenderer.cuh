#pragma once
#include "LaunchParams.hpp"

#include "../info/RenderInfo.hpp"
#include "../info/WorldInfo.hpp"
#include "../info/CameraInfo.hpp"
#include "../abstract/IRenderer.hpp"

template <typename T>
struct SbtRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

class OptixRenderer final : public IRenderer
{
public:
	OptixRenderer(const RenderInfo* render_info, WorldInfo* world_info, const SkyInfo* sky_info, const CameraInfo* camera_info);
	~OptixRenderer() override;

	OptixRenderer(const OptixRenderer&) = delete;
	OptixRenderer(OptixRenderer&&) = delete;
	OptixRenderer& operator=(const OptixRenderer&) = delete;
	OptixRenderer& operator=(OptixRenderer&&) = delete;

	void render() override;
	void refresh_buffer() override;
	void refresh_object(int32_t index) override;
	void refresh_material(int32_t index) const override;
	void refresh_texture(int32_t index) const override;
	void recreate_image() override;
	void recreate_sky() override;
	void map_frame_memory() override;
	void allocate_world() override;
	void deallocate_world() override;

private:
	void init_optix();
	void create_modules();
	void create_programs();
	void create_pipeline();
	void prepare_as(const OptixBuildInput& build_input, void*& buffer, OptixTraversableHandle& handle, OptixBuildOperation operation) const;
	void prepare_gas(Object& object, OptixTraversableHandle& handle, void*& buffer, OptixBuildOperation operation) const;
	void prepare_ias(std::vector<OptixTraversableHandle>& gases, OptixBuildOperation operation);
	void create_sbt();

	const RenderInfo* render_info_ = nullptr;
	WorldInfo* world_info_ = nullptr;
	const SkyInfo* sky_info_ = nullptr;
	const CameraInfo* camera_info_ = nullptr;

	cudaStream_t stream_{};
	OptixDeviceContext context_ = nullptr;
	OptixModule module_ = nullptr;
	OptixPipeline pipeline_ = nullptr;
	OptixShaderBindingTable sbt_{};

	OptixPipelineCompileOptions pipeline_compile_options_{};
	OptixModuleCompileOptions module_compile_options_{};

	std::vector<OptixProgramGroup> raygen_programs_{};
	std::vector<OptixProgramGroup> miss_programs_{};
	std::vector<OptixProgramGroup> hit_programs_{};

	SbtRecord<RayGenData>* d_raygen_records_ = nullptr;
	SbtRecord<MissData>* d_miss_records_ = nullptr;
	SbtRecord<HitGroupData>* d_hit_records_ = nullptr;
	std::vector<void*> gas_buffers_{};
	std::vector<OptixTraversableHandle> gas_handles_{};
	void* ias_buffer_ = nullptr;
	OptixTraversableHandle ias_handle_{};

	uint4* xoshiro_initial_ = nullptr;
	LaunchParams h_launch_params_{};
	LaunchParams* d_launch_params_ = nullptr;
};