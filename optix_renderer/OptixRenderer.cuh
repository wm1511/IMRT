#pragma once
#include "LaunchParams.hpp"
#include "../common/Camera.cuh"
#include "../common/Color.cuh"

#include "../info/RenderInfo.hpp"
#include "../info/WorldInfo.hpp"
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
	OptixRenderer(const RenderInfo* render_info, const WorldInfo* world_info, const SkyInfo* sky_info);
	~OptixRenderer() override;

	OptixRenderer(const OptixRenderer&) = delete;
	OptixRenderer(OptixRenderer&&) = delete;
	OptixRenderer& operator=(const OptixRenderer&) = delete;
	OptixRenderer& operator=(OptixRenderer&&) = delete;

	void render_static() override;
	void render_progressive() override;
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
	OptixTraversableHandle build_as();

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
	SbtRecord<RayGenData>* device_raygen_records_ = nullptr;
	SbtRecord<MissData>* device_miss_records_ = nullptr;
	SbtRecord<HitGroupData>* device_hit_records_ = nullptr;
	float3* device_vertex_buffer_ = nullptr;
	uint3* device_index_buffer_ = nullptr;
	void* device_as_buffer_ = nullptr;

	LaunchParams host_launch_params_{};
	LaunchParams* device_launch_params_ = nullptr;

	cudaStream_t stream_{};
};