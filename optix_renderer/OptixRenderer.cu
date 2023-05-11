// ReSharper disable once CppPrecompiledHeaderIsNotIncluded
#include "OptixRenderer.cuh"
#include "../common/Utils.hpp"

#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#ifdef _DEBUG
static void context_log(unsigned int level, const char* tag, const char* message, void*)
{
	printf("[%u][%s]: %s", level, tag, message);
}
#endif

OptixRenderer::OptixRenderer(const RenderInfo* render_info, const WorldInfo* world_info, const SkyInfo* sky_info)
	: render_info_(render_info), world_info_(world_info), sky_info_(sky_info)
{
	init_optix();
	create_modules();
	create_programs();
	create_pipeline();
	create_sbt();

	CCE(cudaMalloc(reinterpret_cast<void**>(device_launch_params_), sizeof(LaunchParams)));
	CCE(cudaMemcpy(reinterpret_cast<void*>(device_launch_params_), &launch_params_, sizeof(LaunchParams), cudaMemcpyHostToDevice));
}

OptixRenderer::~OptixRenderer()
{
	CCE(cudaStreamDestroy(stream_));
	CCE(cudaFree(reinterpret_cast<void*>(frame_buffer_)));
	CCE(cudaFree(reinterpret_cast<void*>(device_launch_params_)));
	CCE(cudaFree(reinterpret_cast<void*>(raygen_record_buffer_)));
	CCE(cudaFree(reinterpret_cast<void*>(miss_record_buffer_)));
	CCE(cudaFree(reinterpret_cast<void*>(hit_record_buffer_)));

	cudaDeviceReset();
}

void OptixRenderer::render()
{
	if (launch_params_.width == 0 || launch_params_.height == 0) 
		return;

	CCE(cudaMemcpy(reinterpret_cast<void*>(device_launch_params_), &launch_params_, sizeof(LaunchParams), cudaMemcpyHostToDevice));

	COE(optixLaunch(
		pipeline_, stream_,
		device_launch_params_,
		sizeof(LaunchParams),
		&sbt_,
		launch_params_.width,
		launch_params_.height,
		1));

	cudaDeviceSynchronize();                                            
    CCE(cudaGetLastError());
}

void OptixRenderer::refresh_buffer()
{
}

void OptixRenderer::refresh_camera()
{
}

void OptixRenderer::refresh_object(int32_t index) const
{
}

void OptixRenderer::refresh_material(int32_t index) const
{
}

void OptixRenderer::refresh_texture(int32_t index) const
{
}

void OptixRenderer::recreate_camera()
{
}

void OptixRenderer::recreate_image()
{
	// if window minimized
	//if (newSize.x == 0 | newSize.y == 0) return;

	CCE(cudaFree(reinterpret_cast<void*>(frame_buffer_)));
	frame_buffer_ = reinterpret_cast<CUdeviceptr>(fetch_external_memory(render_info_->frame_handle, render_info_->frame_size));

	launch_params_.width = render_info_->width;
	launch_params_.height = render_info_->height;
	launch_params_.frame_buffer = reinterpret_cast<float4*>(frame_buffer_);
}

void OptixRenderer::recreate_sky()
{
}

void OptixRenderer::allocate_world()
{
}

void OptixRenderer::deallocate_world() const
{
}

void OptixRenderer::init_optix()
{
	CCE(cudaFree(nullptr));

	COE(optixInit());

	OptixDeviceContextOptions options{};

#ifdef _DEBUG
	options.logCallbackFunction = &context_log;
	options.logCallbackLevel = 4;
#endif

	const CUcontext cuda_context = nullptr;
	CCE(cudaStreamCreate(&stream_));
	COE(optixDeviceContextCreate(cuda_context, &options, &context_));
}

void OptixRenderer::create_modules()
{
	OptixModuleCompileOptions   module_compile_options{};
	module_compile_options.maxRegisterCount = 50;
	module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

	OptixPipelineCompileOptions pipeline_compile_options{};
	pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	pipeline_compile_options.usesMotionBlur = false;
	pipeline_compile_options.numPayloadValues = 2;
	pipeline_compile_options.numAttributeValues = 2;
	pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	pipeline_compile_options.pipelineLaunchParamsVariableName = "launch_params";

	const std::string ptx = read_ptx("OptixPrograms");

	COE(optixModuleCreate(
		context_,
		&module_compile_options,
		&pipeline_compile_options,
		ptx.c_str(),
		ptx.size(),
		nullptr, nullptr,
		&module_));
}

void OptixRenderer::create_programs()
{
	raygen_programs_.resize(1);
	OptixProgramGroupOptions rg_options = {};
	OptixProgramGroupDesc rg_desc = {};
	rg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	rg_desc.raygen.module = module_;
	rg_desc.raygen.entryFunctionName = "__raygen__render";

	COE(optixProgramGroupCreate(
		context_,
		&rg_desc,
		1,
		&rg_options,
		nullptr, nullptr,
		raygen_programs_.data()));

	miss_programs_.resize(1);
	OptixProgramGroupOptions m_options = {};
	OptixProgramGroupDesc m_desc = {};
	m_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	m_desc.miss.module = module_;
	m_desc.miss.entryFunctionName = "__miss__radiance";

	COE(optixProgramGroupCreate(
		context_,
		&m_desc,
		1,
		&m_options,
		nullptr, nullptr,
		miss_programs_.data()));

	hit_programs_.resize(1);
	OptixProgramGroupOptions hg_options = {};
	OptixProgramGroupDesc hg_desc = {};
	hg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	hg_desc.hitgroup.moduleCH = module_;
	hg_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
	hg_desc.hitgroup.moduleAH = module_;
	hg_desc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

	COE(optixProgramGroupCreate(
		context_,
		&hg_desc,
		1,
		&hg_options,
		nullptr, nullptr,
		hit_programs_.data()));
}

void OptixRenderer::create_pipeline()
{
	std::vector<OptixProgramGroup> program_groups;
	for (auto pg : raygen_programs_)
		program_groups.push_back(pg);
	for (auto pg : miss_programs_)
		program_groups.push_back(pg);
	for (auto pg : hit_programs_)
		program_groups.push_back(pg);

	constexpr OptixPipelineCompileOptions pipeline_compile_options{};
	constexpr OptixPipelineLinkOptions pipeline_link_options{};

	COE(optixPipelineCreate(
		context_,
		&pipeline_compile_options,
		&pipeline_link_options,
		program_groups.data(),
		static_cast<uint32_t>(program_groups.size()),
		nullptr, nullptr,
		&pipeline_));

	COE(optixPipelineSetStackSize
	(/* [in] The pipeline to configure the stack size for */
		pipeline_,
		/* [in] The direct stack size requirement for direct
		   callables invoked from IS or AH. */
		2 * 1024,
		/* [in] The direct stack size requirement for direct
		   callables invoked from RG, MS, or CH.  */
		2 * 1024,
		/* [in] The continuation stack requirement. */
		2 * 1024,
		/* [in] The maximum depth of a traversable graph
		   passed to trace. */
		1));
}

void OptixRenderer::create_sbt()
{
	std::vector<SbtRecord<RayGenData>> raygen_records;
	for (const auto& raygen_program : raygen_programs_)
	{
		SbtRecord<RayGenData> rec{};
		COE(optixSbtRecordPackHeader(raygen_program, &rec));
		raygen_records.push_back(rec);
	}

	CCE(cudaMalloc(reinterpret_cast<void**>(raygen_record_buffer_), raygen_records.size() * sizeof(SbtRecord<RayGenData>)));
	CCE(cudaMemcpy(reinterpret_cast<void*>(raygen_record_buffer_), raygen_records.data(), raygen_records.size() * sizeof(SbtRecord<RayGenData>), cudaMemcpyHostToDevice));

	sbt_.raygenRecord = raygen_record_buffer_;

	std::vector<SbtRecord<MissData>> miss_records;
	for (const auto& miss_program : miss_programs_)
	{
		SbtRecord<MissData> rec{};
		COE(optixSbtRecordPackHeader(miss_program, &rec));
		miss_records.push_back(rec);
	}

	CCE(cudaMalloc(reinterpret_cast<void**>(miss_record_buffer_), miss_records.size() * sizeof(SbtRecord<MissData>)));
	CCE(cudaMemcpy(reinterpret_cast<void*>(miss_record_buffer_), miss_records.data(), miss_records.size() * sizeof(SbtRecord<MissData>), cudaMemcpyHostToDevice));

	sbt_.missRecordBase = miss_record_buffer_;
	sbt_.missRecordStrideInBytes = sizeof(SbtRecord<MissData>);
	sbt_.missRecordCount = static_cast<uint32_t>(miss_records.size());

	constexpr int32_t num_objects = 1;
	std::vector<SbtRecord<HitGroupData>> hitgroup_records;
	for (int i = 0; i < num_objects; i++)
	{
		constexpr int32_t objectType = 0;
		SbtRecord<HitGroupData> rec{};
		COE(optixSbtRecordPackHeader(hit_programs_[objectType], &rec));
		rec.data.object_id = i;
		hitgroup_records.push_back(rec);
	}

	CCE(cudaMalloc(reinterpret_cast<void**>(hit_record_buffer_), hitgroup_records.size() * sizeof(SbtRecord<HitGroupData>)));
	CCE(cudaMemcpy(reinterpret_cast<void*>(hit_record_buffer_), hitgroup_records.data(), hitgroup_records.size() * sizeof(SbtRecord<HitGroupData>), cudaMemcpyHostToDevice));

	sbt_.hitgroupRecordBase = hit_record_buffer_;
	sbt_.hitgroupRecordStrideInBytes = sizeof(SbtRecord<HitGroupData>);
	sbt_.hitgroupRecordCount = static_cast<uint32_t>(hitgroup_records.size());
}
