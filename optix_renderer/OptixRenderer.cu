// ReSharper disable once CppPrecompiledHeaderIsNotIncluded
#include "OptixRenderer.cuh"

#include "../common/Utils.hpp"
#include "../cuda_renderer/CudaKernels.cuh"

#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#ifdef _DEBUG
static void context_log(unsigned int level, const char* tag, const char* message, void*)
{
	printf("[%u][%s]: %s\n", level, tag, message);
}
#endif

OptixRenderer::OptixRenderer(const RenderInfo* render_info, const WorldInfo* world_info, const SkyInfo* sky_info, const CameraInfo* camera_info)
	: render_info_(render_info), world_info_(world_info), sky_info_(sky_info), camera_info_(camera_info)
{
	const uint32_t width = render_info_->width;
	const uint32_t height = render_info_->height;
	constexpr int32_t thread_x = 16;
	constexpr int32_t thread_y = 16;
	auto blocks = dim3((width + thread_x - 1) / thread_x, (height + thread_y - 1) / thread_y);
	auto threads = dim3(thread_x, thread_y);

	init_optix();
	create_modules();
	create_programs();
	h_launch_params_.traversable = build_as();
	create_pipeline();
	create_sbt();

	CCE(cudaMalloc(reinterpret_cast<void**>(&h_launch_params_.accumulation_buffer), sizeof(float4) * width * height));
	CCE(cudaMalloc(reinterpret_cast<void**>(&xoshiro_initial_), sizeof(uint4) * width * height));
	CCE(cudaMalloc(reinterpret_cast<void**>(&h_launch_params_.xoshiro_state), sizeof(uint4) * width * height));

	random_init<<<blocks, threads>>>(width, height, xoshiro_initial_);
	CCE(cudaGetLastError());
	CCE(cudaDeviceSynchronize());

	CCE(cudaMemcpy(h_launch_params_.xoshiro_state, xoshiro_initial_, sizeof(uint4) * width * height, cudaMemcpyDeviceToDevice));

	if (sky_info_->h_hdr_data)
	{
		const uint64_t hdr_size = sizeof(float3) * sky_info_->hdr_width * sky_info_->hdr_height;
		CCE(cudaMalloc(reinterpret_cast<void**>(&sky_info_->d_hdr_data), hdr_size));
		CCE(cudaMemcpy(sky_info_->d_hdr_data, sky_info_->h_hdr_data, hdr_size, cudaMemcpyHostToDevice));
	}

	h_launch_params_.sky_info = *sky_info;
	h_launch_params_.camera_info = *camera_info;

	CCE(cudaMalloc(reinterpret_cast<void**>(&d_launch_params_), sizeof(LaunchParams)));
	CCE(cudaMemcpy(d_launch_params_, &h_launch_params_, sizeof(LaunchParams), cudaMemcpyHostToDevice));
}

OptixRenderer::~OptixRenderer()
{
	if (sky_info_->h_hdr_data)
		CCE(cudaFree(sky_info_->d_hdr_data));

	CCE(cudaFree(h_launch_params_.xoshiro_state));
	CCE(cudaFree(xoshiro_initial_));
	CCE(cudaFree(h_launch_params_.accumulation_buffer));

	CCE(cudaStreamDestroy(stream_));
	CCE(cudaFree(d_launch_params_));
	CCE(cudaFree(d_raygen_records_));
	CCE(cudaFree(d_miss_records_));
	CCE(cudaFree(d_hit_records_));
	CCE(cudaFree(d_index_buffer_));
	CCE(cudaFree(d_vertex_buffer_));
	CCE(cudaFree(d_as_buffer_));

	cudaDeviceReset();
}

void OptixRenderer::render_static()
{
	//if (render_info_)
}

void OptixRenderer::render_progressive()
{
	const auto frame_buffer = static_cast<float4*>(fetch_external_memory(render_info_->frame_handle, render_info_->frame_size));

	h_launch_params_.width = render_info_->width;
	h_launch_params_.height = render_info_->height;
	h_launch_params_.frames_since_refresh = render_info_->frames_since_refresh;
	h_launch_params_.frame_buffer = frame_buffer;
	h_launch_params_.sky_info = *sky_info_;
	h_launch_params_.camera_info = *camera_info_;

	CCE(cudaMemcpy(d_launch_params_, &h_launch_params_, sizeof(LaunchParams), cudaMemcpyHostToDevice));

	COE(optixLaunch(
		pipeline_, stream_,
		reinterpret_cast<CUdeviceptr>(d_launch_params_),
		sizeof(LaunchParams),
		&sbt_,
		h_launch_params_.width,
		h_launch_params_.height,
		1));

	CCE(cudaDeviceSynchronize());
	CCE(cudaGetLastError());

	CCE(cudaFree(frame_buffer));
}

void OptixRenderer::refresh_buffer()
{
	const uint32_t width = render_info_->width;
	const uint32_t height = render_info_->height;

	CCE(cudaMemset(h_launch_params_.accumulation_buffer, 0, sizeof(float4) * width * height));
	CCE(cudaMemcpy(h_launch_params_.xoshiro_state, xoshiro_initial_, sizeof(uint4) * width * height, cudaMemcpyDeviceToDevice));
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

void OptixRenderer::recreate_image()
{
	const uint32_t width = render_info_->width;
	const uint32_t height = render_info_->height;
	constexpr int32_t thread_x = 16;
	constexpr int32_t thread_y = 16;
	auto blocks = dim3((width + thread_x - 1) / thread_x, (height + thread_y - 1) / thread_y);
	auto threads = dim3(thread_x, thread_y);

	CCE(cudaFree(h_launch_params_.xoshiro_state));
	CCE(cudaFree(xoshiro_initial_));
	CCE(cudaFree(h_launch_params_.accumulation_buffer));
	CCE(cudaMalloc(reinterpret_cast<void**>(&h_launch_params_.accumulation_buffer), sizeof(float4) * width * height));
	CCE(cudaMalloc(reinterpret_cast<void**>(&xoshiro_initial_), sizeof(uint4) * width * height));
	CCE(cudaMalloc(reinterpret_cast<void**>(&h_launch_params_.xoshiro_state), sizeof(uint4) * width * height));

	random_init<<<blocks, threads>>>(width, height, xoshiro_initial_);
	CCE(cudaGetLastError());
	CCE(cudaDeviceSynchronize());
}

void OptixRenderer::recreate_sky()
{
	CCE(cudaFree(sky_info_->d_hdr_data));

	if (sky_info_->h_hdr_data)
	{
		const uint64_t hdr_size = sizeof(float3) * sky_info_->hdr_width * sky_info_->hdr_height;
		CCE(cudaMalloc(reinterpret_cast<void**>(&sky_info_->d_hdr_data), hdr_size));
		CCE(cudaMemcpy(sky_info_->d_hdr_data, sky_info_->h_hdr_data, hdr_size, cudaMemcpyHostToDevice));
	}
	else
		sky_info_->d_hdr_data = nullptr;
}

void OptixRenderer::map_frame_memory()
{
	const auto frame_buffer = static_cast<float4*>(fetch_external_memory(render_info_->frame_handle, render_info_->frame_size));

	CCE(cudaMemcpy(render_info_->frame_data, frame_buffer, render_info_->frame_size, cudaMemcpyDeviceToHost));
	CCE(cudaFree(frame_buffer));
}

void OptixRenderer::allocate_world()
{
}

void OptixRenderer::deallocate_world() const
{
}

void OptixRenderer::init_optix()
{
	COE(optixInit());

	OptixDeviceContextOptions options{};

#ifdef _DEBUG
	options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
	options.logCallbackFunction = &context_log;
	options.logCallbackLevel = 4;
#endif

	const CUcontext cuda_context = nullptr;

	CCE(cudaStreamCreate(&stream_));
	COE(optixDeviceContextCreate(cuda_context, &options, &context_));
}

void OptixRenderer::create_modules()
{
	OptixModuleCompileOptions module_compile_options{};
	module_compile_options.maxRegisterCount = 50;
#ifdef _DEBUG
	module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
	module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
	module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif

	OptixPipelineCompileOptions pipeline_compile_options{};
	pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	pipeline_compile_options.usesMotionBlur = false;
	pipeline_compile_options.numPayloadValues = 2;
	pipeline_compile_options.numAttributeValues = 2;
	pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	pipeline_compile_options.pipelineLaunchParamsVariableName = "launch_params";

	const std::string shader = read_shader("OptixPrograms.optixir");

	COE(optixModuleCreate(
		context_,
		&module_compile_options,
		&pipeline_compile_options,
		shader.c_str(),
		shader.size(),
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

	program_groups.reserve(program_groups.size() + raygen_programs_.size());
	for (auto pg : raygen_programs_)
		program_groups.push_back(pg);

	program_groups.reserve(program_groups.size() + miss_programs_.size());
	for (auto pg : miss_programs_)
		program_groups.push_back(pg);

	program_groups.reserve(program_groups.size() + hit_programs_.size());
	for (auto pg : hit_programs_)
		program_groups.push_back(pg);

	OptixPipelineCompileOptions pipeline_compile_options{};
	pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	pipeline_compile_options.usesMotionBlur = false;
	pipeline_compile_options.numPayloadValues = 2;
	pipeline_compile_options.numAttributeValues = 2;
#ifdef _DEBUG
	pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG;
#else
	pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
	pipeline_compile_options.pipelineLaunchParamsVariableName = "launch_params";

	const OptixPipelineLinkOptions pipeline_link_options{ static_cast<uint32_t>(render_info_->max_depth < 32 ? render_info_->max_depth : 31) };

	COE(optixPipelineCreate(
		context_,
		&pipeline_compile_options,
		&pipeline_link_options,
		program_groups.data(),
		static_cast<uint32_t>(program_groups.size()),
		nullptr, nullptr,
		&pipeline_));

	COE(optixPipelineSetStackSize(pipeline_, 2 * 1024, 2 * 1024, 2 * 1024, 1));
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

	CCE(cudaMalloc(reinterpret_cast<void**>(&d_raygen_records_), raygen_records.size() * sizeof(SbtRecord<RayGenData>)));
	CCE(cudaMemcpy(d_raygen_records_, raygen_records.data(), raygen_records.size() * sizeof(SbtRecord<RayGenData>), cudaMemcpyHostToDevice));

	sbt_.raygenRecord = reinterpret_cast<CUdeviceptr>(d_raygen_records_);

	std::vector<SbtRecord<MissData>> miss_records;
	for (const auto& miss_program : miss_programs_)
	{
		SbtRecord<MissData> rec{};
		COE(optixSbtRecordPackHeader(miss_program, &rec));
		miss_records.push_back(rec);
	}

	CCE(cudaMalloc(reinterpret_cast<void**>(&d_miss_records_), miss_records.size() * sizeof(SbtRecord<MissData>)));
	CCE(cudaMemcpy(d_miss_records_, miss_records.data(), miss_records.size() * sizeof(SbtRecord<MissData>), cudaMemcpyHostToDevice));

	sbt_.missRecordBase = reinterpret_cast<CUdeviceptr>(d_miss_records_);
	sbt_.missRecordStrideInBytes = sizeof(SbtRecord<MissData>);
	sbt_.missRecordCount = static_cast<uint32_t>(miss_records.size());

	constexpr int32_t num_objects = 1;
	std::vector<SbtRecord<HitGroupData>> hitgroup_records;
	for (int i = 0; i < num_objects; i++)
	{
		constexpr int32_t object_type = 0;
		SbtRecord<HitGroupData> rec{};
		COE(optixSbtRecordPackHeader(hit_programs_[object_type], &rec));
		rec.data.vertex = d_vertex_buffer_;
		rec.data.index = d_index_buffer_;
		rec.data.color = make_float3(0.5f);
		hitgroup_records.push_back(rec);
	}

	CCE(cudaMalloc(reinterpret_cast<void**>(&d_hit_records_), hitgroup_records.size() * sizeof(SbtRecord<HitGroupData>)));
	CCE(cudaMemcpy(d_hit_records_, hitgroup_records.data(), hitgroup_records.size() * sizeof(SbtRecord<HitGroupData>), cudaMemcpyHostToDevice));

	sbt_.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(d_hit_records_);
	sbt_.hitgroupRecordStrideInBytes = sizeof(SbtRecord<HitGroupData>);
	sbt_.hitgroupRecordCount = static_cast<uint32_t>(hitgroup_records.size());
}

OptixTraversableHandle OptixRenderer::build_as()
{
	std::vector<float> vertices
	{
		-1, -1,  1,
		 1, -1,  1,
		-1,  1,  1,
		 1,  1,  1,
		-1, -1, -1,
		 1, -1, -1,
		-1,  1, -1,
		 1,  1, -1
	};

	std::vector<uint32_t> indices
	{
		2, 6, 7,
		2, 3, 7,
		0, 4, 5,
		0, 1, 5,
		0, 2, 6,
		0, 4, 6,
		1, 3, 7,
		1, 5, 7,
		0, 2, 3,
		0, 1, 3,
		4, 6, 7,
		4, 5, 7
	};

	CCE(cudaMalloc(reinterpret_cast<void**>(&d_vertex_buffer_), vertices.size() * sizeof(float)));
	CCE(cudaMemcpy(d_vertex_buffer_, vertices.data(), vertices.size() * sizeof(float), cudaMemcpyHostToDevice));
	CCE(cudaMalloc(reinterpret_cast<void**>(&d_index_buffer_), indices.size() * sizeof(uint32_t)));
	CCE(cudaMemcpy(d_index_buffer_, indices.data(), indices.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

	OptixTraversableHandle as_handle{ 0 };

	OptixBuildInput triangle_input = {};
	triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

	triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	triangle_input.triangleArray.vertexStrideInBytes = sizeof(float3);
	triangle_input.triangleArray.numVertices = static_cast<uint32_t>(vertices.size());
	triangle_input.triangleArray.vertexBuffers = reinterpret_cast<CUdeviceptr*>(&d_vertex_buffer_);

	triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	triangle_input.triangleArray.indexStrideInBytes = sizeof(int3);
	triangle_input.triangleArray.numIndexTriplets = static_cast<uint32_t>(indices.size());
	triangle_input.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(d_index_buffer_);

	uint32_t triangle_input_flags[1] = { 0 };

	triangle_input.triangleArray.flags = triangle_input_flags;
	triangle_input.triangleArray.numSbtRecords = 1;
	triangle_input.triangleArray.sbtIndexOffsetBuffer = 0;
	triangle_input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
	triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = 0;

	OptixAccelBuildOptions accel_options = {};
	accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accel_options.motionOptions.numKeys = 1;
	accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes blas_buffer_sizes;
	COE(optixAccelComputeMemoryUsage(
		context_,
		&accel_options,
		&triangle_input,
		1,
		&blas_buffer_sizes));

	uint64_t* compacted_size_buffer = nullptr;
	CCE(cudaMalloc(reinterpret_cast<void**>(&compacted_size_buffer), sizeof(uint64_t)));

	OptixAccelEmitDesc emit_desc;
	emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emit_desc.result = reinterpret_cast<CUdeviceptr>(compacted_size_buffer);

	void* temp_buffer = nullptr;
	CCE(cudaMalloc(&temp_buffer, blas_buffer_sizes.tempSizeInBytes));

	void* output_buffer;
	CCE(cudaMalloc(&output_buffer, blas_buffer_sizes.outputSizeInBytes));

	COE(optixAccelBuild(context_,
		nullptr,
		&accel_options,
		&triangle_input,
		1,
		reinterpret_cast<CUdeviceptr>(temp_buffer),
		blas_buffer_sizes.tempSizeInBytes,
		reinterpret_cast<CUdeviceptr>(output_buffer),
		blas_buffer_sizes.outputSizeInBytes,
		&as_handle,
		&emit_desc, 1));

	CCE(cudaDeviceSynchronize());
	CCE(cudaGetLastError());

	uint64_t compacted_size;
	CCE(cudaMemcpy(&compacted_size, compacted_size_buffer, sizeof(uint64_t), cudaMemcpyDeviceToHost));

	CCE(cudaMalloc(&d_as_buffer_, compacted_size));
	COE(optixAccelCompact(
		context_,
		nullptr,
		as_handle,
		reinterpret_cast<CUdeviceptr>(d_as_buffer_),
		compacted_size,
		&as_handle));

	CCE(cudaDeviceSynchronize());
	CCE(cudaGetLastError());

	CCE(cudaFree(output_buffer));
	CCE(cudaFree(temp_buffer));
	CCE(cudaFree(compacted_size_buffer));

	return as_handle;
}
