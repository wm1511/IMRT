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

OptixRenderer::OptixRenderer(const RenderInfo* render_info, WorldInfo* world_info, const SkyInfo* sky_info, const CameraInfo* camera_info)
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
	create_pipeline();

	allocate_world();

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

	CCE(cudaMalloc(reinterpret_cast<void**>(&d_launch_params_), sizeof(LaunchParams)));
	CCE(cudaMemcpy(d_launch_params_, &h_launch_params_, sizeof(LaunchParams), cudaMemcpyHostToDevice));
}

OptixRenderer::~OptixRenderer()
{
	if (sky_info_->h_hdr_data)
		CCE(cudaFree(sky_info_->d_hdr_data));

	deallocate_world();

	CCE(cudaFree(h_launch_params_.xoshiro_state));
	CCE(cudaFree(xoshiro_initial_));
	CCE(cudaFree(h_launch_params_.accumulation_buffer));

	CCE(cudaFree(d_launch_params_));

	CCE(cudaStreamDestroy(stream_));
	COE(optixDeviceContextDestroy(context_));

	cudaDeviceReset();
}

void OptixRenderer::render()
{
	const auto frame_buffer = static_cast<float4*>(fetch_external_memory(render_info_->frame_handle, render_info_->frame_size));

	h_launch_params_.width = render_info_->width;
	h_launch_params_.height = render_info_->height;
	h_launch_params_.frame_buffer = frame_buffer;
	h_launch_params_.sky_info = *sky_info_;
	h_launch_params_.camera_info = *camera_info_;

	if (render_info_->progressive)
	{
		h_launch_params_.sampling_denominator = render_info_->frames_since_refresh;

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
	}
	else
	{
		CCE(cudaMemset(frame_buffer, 0, render_info_->frame_size));
		h_launch_params_.sampling_denominator = render_info_->samples_per_pixel;

		CCE(cudaMemcpy(d_launch_params_, &h_launch_params_, sizeof(LaunchParams), cudaMemcpyHostToDevice));

		for (uint32_t i = 0; i < h_launch_params_.sampling_denominator; i++)
		{
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
		}
	}

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

void OptixRenderer::refresh_material(const int32_t index) const
{
	const Object& object = world_info_->objects_[index];

	SbtRecord<HitGroupData> rec{};
	COE(optixSbtRecordPackHeader(hit_programs_[enum_cast(object.type) - 1], &rec));
	rec.data.texture = world_info_->textures_[object.texture_id];
	rec.data.material = world_info_->materials_[object.material_id];
	rec.data.object = object;

	CCE(cudaMemcpy(d_hit_records_ + index, &rec, sizeof(SbtRecord<HitGroupData>), cudaMemcpyHostToDevice));
}

void OptixRenderer::refresh_texture(const int32_t index) const
{
	const Object& object = world_info_->objects_[index];

	SbtRecord<HitGroupData> rec{};
	COE(optixSbtRecordPackHeader(hit_programs_[enum_cast(object.type) - 1], &rec));
	rec.data.texture = world_info_->textures_[object.texture_id];
	rec.data.material = world_info_->materials_[object.material_id];
	rec.data.object = object;

	CCE(cudaMemcpy(d_hit_records_ + index, &rec, sizeof(SbtRecord<HitGroupData>), cudaMemcpyHostToDevice));
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
	for (auto& texture : world_info_->textures_)
	{
		if (texture.type == TextureType::IMAGE)
		{
			const auto image_data = &texture.image;
			const uint64_t image_size = sizeof(float) * image_data->width * image_data->height * 3;

			CCE(cudaMalloc(reinterpret_cast<void**>(&image_data->d_data), image_size));
			CCE(cudaMemcpy(image_data->d_data, image_data->h_data, image_size, cudaMemcpyHostToDevice));
		}
	}

	for (auto& object : world_info_->objects_)
	{
		if (object.type == ObjectType::MODEL)
		{
			const auto model_data = &object.model;

			CCE(cudaMalloc(reinterpret_cast<void**>(&model_data->d_vertices), model_data->vertex_count * sizeof(float3)));
			CCE(cudaMemcpy(model_data->d_vertices, model_data->h_vertices, model_data->vertex_count * sizeof(float3), cudaMemcpyHostToDevice));

			CCE(cudaMalloc(reinterpret_cast<void**>(&model_data->d_indices), model_data->index_count * sizeof(uint3)));
			CCE(cudaMemcpy(model_data->d_indices, model_data->h_indices, model_data->index_count * sizeof(uint3), cudaMemcpyHostToDevice));

			CCE(cudaMalloc(reinterpret_cast<void**>(&model_data->d_normals), model_data->vertex_count * sizeof(float3)));
			CCE(cudaMemcpy(model_data->d_normals, model_data->h_normals, model_data->vertex_count * sizeof(float3), cudaMemcpyHostToDevice));

			CCE(cudaMalloc(reinterpret_cast<void**>(&model_data->d_uv), model_data->vertex_count * sizeof(float2)));
			CCE(cudaMemcpy(model_data->d_uv, model_data->h_uv, model_data->vertex_count * sizeof(float2), cudaMemcpyHostToDevice));
		}
	}

	std::vector<OptixTraversableHandle> gases{};

	build_gases(gases, gas_buffers_);
	h_launch_params_.traversable = build_ias(gases);

	create_sbt();
}

void OptixRenderer::deallocate_world()
{
	CCE(cudaFree(d_raygen_records_));
	CCE(cudaFree(d_miss_records_));
	CCE(cudaFree(d_hit_records_));

	CCE(cudaFree(ias_buffer_));

	for (const auto& buffer : gas_buffers_)
		CCE(cudaFree(buffer));

	gas_buffers_.clear();

	for (const auto& object : world_info_->objects_)
	{
		if (object.type == ObjectType::MODEL)
		{
			CCE(cudaFree(object.model.d_vertices));
			CCE(cudaFree(object.model.d_indices));
			CCE(cudaFree(object.model.d_normals));
			CCE(cudaFree(object.model.d_uv));
		}
	}

	for (const auto& texture : world_info_->textures_)
	{
		if (texture.type == TextureType::IMAGE)
			CCE(cudaFree(texture.image.d_data));
	}
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
	module_compile_options_.maxRegisterCount = 50;
#ifdef _DEBUG
	module_compile_options_.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
	module_compile_options_.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
	module_compile_options_.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	module_compile_options_.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif

	pipeline_compile_options_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
	pipeline_compile_options_.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM |
		OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;
	pipeline_compile_options_.usesMotionBlur = false;
	pipeline_compile_options_.numPayloadValues = 2;
	pipeline_compile_options_.numAttributeValues = 2;
#ifdef _DEBUG
	pipeline_compile_options_.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG;
#else
	pipeline_compile_options_.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
	pipeline_compile_options_.pipelineLaunchParamsVariableName = "launch_params";

	const std::string shader = read_shader("OptixPrograms.optixir");

	COE(optixModuleCreate(
		context_,
		&module_compile_options_,
		&pipeline_compile_options_,
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

	hit_programs_.resize(3);
	OptixProgramGroupOptions hg_options = {};
	OptixModule sphere_is_module{};
	OptixBuiltinISOptions sphere_is_options = {};
	sphere_is_options.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
	sphere_is_options.usesMotionBlur = 0;

	COE(optixBuiltinISModuleGet(
		context_, 
		&module_compile_options_, 
		&pipeline_compile_options_, 
		&sphere_is_options, 
		&sphere_is_module));

	OptixProgramGroupDesc s_hg_desc = {};
	s_hg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	s_hg_desc.hitgroup.moduleCH = module_;
	s_hg_desc.hitgroup.entryFunctionNameCH = "__closesthit__sphere";
	s_hg_desc.hitgroup.moduleIS = sphere_is_module;

	COE(optixProgramGroupCreate(
		context_,
		&s_hg_desc,
		1,
		&hg_options,
		nullptr, nullptr,
		hit_programs_.data()));

	OptixProgramGroupDesc c_hg_desc = {};
	c_hg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	c_hg_desc.hitgroup.moduleCH = module_;
	c_hg_desc.hitgroup.entryFunctionNameCH = "__closesthit__cylinder";
	c_hg_desc.hitgroup.moduleIS = module_;
	c_hg_desc.hitgroup.entryFunctionNameIS = "__intersection__cylinder";

	COE(optixProgramGroupCreate(
		context_,
		&c_hg_desc,
		1,
		&hg_options,
		nullptr, nullptr,
		hit_programs_.data() + 1));

	OptixProgramGroupDesc t_hg_desc = {};
	t_hg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	t_hg_desc.hitgroup.moduleCH = module_;
	t_hg_desc.hitgroup.entryFunctionNameCH = "__closesthit__triangle";

	COE(optixProgramGroupCreate(
		context_,
		&t_hg_desc,
		1,
		&hg_options,
		nullptr, nullptr,
		hit_programs_.data() + 2));
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

	const OptixPipelineLinkOptions pipeline_link_options{ static_cast<uint32_t>(render_info_->max_depth) };

	COE(optixPipelineCreate(
		context_,
		&pipeline_compile_options_,
		&pipeline_link_options,
		program_groups.data(),
		static_cast<uint32_t>(program_groups.size()),
		nullptr, nullptr,
		&pipeline_));

	COE(optixPipelineSetStackSize(pipeline_, 2 * 1024, 2 * 1024, 2 * 1024, 2));
}

void OptixRenderer::build_gases(std::vector<OptixTraversableHandle>& gases, std::vector<void*>& buffers) const
{
	for (auto& object : world_info_->objects_)
	{
		gases.resize(gases.size() + 1);
		buffers.resize(buffers.size() + 1);

		OptixBuildInput input{};
		uint32_t flags[1] = { OPTIX_BUILD_FLAG_NONE };

		if (object.type == ObjectType::SPHERE)
		{
			input.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;

			float3* d_center;
			float* d_radius;

			CCE(cudaMalloc(reinterpret_cast<void**>(&d_center), sizeof(float3)));
			CCE(cudaMemcpy(d_center, &object.sphere.center, sizeof(float3), cudaMemcpyHostToDevice));
			CCE(cudaMalloc(reinterpret_cast<void**>(&d_radius), sizeof(float)));
			CCE(cudaMemcpy(d_radius, &object.sphere.radius, sizeof(float), cudaMemcpyHostToDevice));

			input.sphereArray.vertexStrideInBytes = sizeof(float3);
			input.sphereArray.numVertices = 1;
			input.sphereArray.vertexBuffers = reinterpret_cast<CUdeviceptr*>(&d_center);

			input.sphereArray.radiusStrideInBytes = sizeof(float);
			input.sphereArray.radiusBuffers = reinterpret_cast<CUdeviceptr*>(&d_radius);

			input.sphereArray.flags = flags;
			input.sphereArray.numSbtRecords = 1;
			input.sphereArray.sbtIndexOffsetBuffer = 0;
			input.sphereArray.sbtIndexOffsetSizeInBytes = 0;
			input.sphereArray.sbtIndexOffsetStrideInBytes = 0;

			build_gas(input, buffers[buffers.size() - 1], gases[gases.size() - 1]);

			CCE(cudaFree(d_center));
			CCE(cudaFree(d_radius));
		}
		else if (object.type == ObjectType::CYLINDER)
		{
			input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

			OptixAabb* d_aabb;

			Boundary boundary = object.cylinder.bound();
			OptixAabb aabb{boundary.min_.x, boundary.min_.y, boundary.min_.z,
					boundary.max_.x, boundary.max_.x, boundary.max_.x};

			CCE(cudaMalloc(reinterpret_cast<void**>(&d_aabb), sizeof(OptixAabb)));
			CCE(cudaMemcpy(d_aabb, &aabb, sizeof(OptixAabb), cudaMemcpyHostToDevice));

			input.customPrimitiveArray.aabbBuffers = reinterpret_cast<CUdeviceptr*>(&d_aabb);
			input.customPrimitiveArray.numPrimitives = 1;

			input.customPrimitiveArray.flags = flags;
			input.customPrimitiveArray.numSbtRecords = 1;
			input.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
			input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
			input.customPrimitiveArray.sbtIndexOffsetStrideInBytes = 0; 
			input.customPrimitiveArray.primitiveIndexOffset = 0;

			build_gas(input, buffers[buffers.size() - 1], gases[gases.size() - 1]);

			CCE(cudaFree(d_aabb));
		}
		else if (object.type == ObjectType::MODEL)
		{
			input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

			input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
			input.triangleArray.vertexStrideInBytes = sizeof(float3);
			input.triangleArray.numVertices = static_cast<uint32_t>(object.model.vertex_count);
			input.triangleArray.vertexBuffers = reinterpret_cast<CUdeviceptr*>(&object.model.d_vertices);

			input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
			input.triangleArray.indexStrideInBytes = sizeof(int3);
			input.triangleArray.numIndexTriplets = static_cast<uint32_t>(object.model.index_count);
			input.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(object.model.d_indices);

			input.triangleArray.flags = flags;
			input.triangleArray.numSbtRecords = 1;
			input.triangleArray.sbtIndexOffsetBuffer = 0;
			input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
			input.triangleArray.sbtIndexOffsetStrideInBytes = 0;

			build_gas(input, buffers[buffers.size() - 1], gases[gases.size() - 1]);
		}
	}
}

void OptixRenderer::build_gas(const OptixBuildInput& build_input, void*& buffer, OptixTraversableHandle& handle) const
{
	OptixAccelBuildOptions accel_options = {};
	accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
	accel_options.motionOptions.numKeys = 1;
	accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes gas_buffer_sizes;
	COE(optixAccelComputeMemoryUsage(
		context_,
		&accel_options,
		&build_input,
		1,
		&gas_buffer_sizes));

	uint64_t* compacted_size_buffer;
	CCE(cudaMalloc(reinterpret_cast<void**>(&compacted_size_buffer), sizeof(uint64_t)));

	OptixAccelEmitDesc emit_desc;
	emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emit_desc.result = reinterpret_cast<CUdeviceptr>(compacted_size_buffer);

	void* temp_buffer = nullptr;
	CCE(cudaMalloc(&temp_buffer, gas_buffer_sizes.tempSizeInBytes));

	void* output_buffer;
	CCE(cudaMalloc(&output_buffer, gas_buffer_sizes.outputSizeInBytes));

	COE(optixAccelBuild(
		context_,
		nullptr,
		&accel_options,
		&build_input,
		1,
		reinterpret_cast<CUdeviceptr>(temp_buffer),
		gas_buffer_sizes.tempSizeInBytes,
		reinterpret_cast<CUdeviceptr>(output_buffer),
		gas_buffer_sizes.outputSizeInBytes,
		&handle,
		&emit_desc, 1));

	CCE(cudaDeviceSynchronize());
	CCE(cudaGetLastError());

	uint64_t compacted_size;
	CCE(cudaMemcpy(&compacted_size, compacted_size_buffer, sizeof(uint64_t), cudaMemcpyDeviceToHost));

	CCE(cudaMalloc(&buffer, compacted_size));
	COE(optixAccelCompact(
		context_,
		nullptr,
		handle,
		reinterpret_cast<CUdeviceptr>(buffer),
		compacted_size,
		&handle));

	CCE(cudaDeviceSynchronize());
	CCE(cudaGetLastError());

	CCE(cudaFree(output_buffer));
	CCE(cudaFree(temp_buffer));
	CCE(cudaFree(compacted_size_buffer));
}

OptixTraversableHandle OptixRenderer::build_ias(std::vector<OptixTraversableHandle>& gases)
{
	OptixTraversableHandle as_handle{};
    std::vector<OptixInstance> instances;

	float transform[12] = {	1,0,0,0,
							0,1,0,0,
							0,0,1,0 };

	for (uint32_t i = 0; i < gases.size(); i++)
    {
        OptixInstance instance = {};
        
        instance.instanceId = 0;
        instance.visibilityMask = 255;
        instance.sbtOffset = i;
        instance.flags = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;
        memcpy_s(instance.transform, sizeof(float) * 12, transform, sizeof transform);
        instance.traversableHandle = gases[i];
        instances.push_back(instance);
    }

    OptixInstance* instance_buffer;
	CCE(cudaMalloc(reinterpret_cast<void**>(&instance_buffer), instances.size() * sizeof(OptixInstance)));
	CCE(cudaMemcpy(instance_buffer, instances.data(), instances.size() * sizeof(OptixInstance), cudaMemcpyHostToDevice));

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    build_input.instanceArray.instances = reinterpret_cast<CUdeviceptr>(instance_buffer);
    build_input.instanceArray.numInstances = static_cast<uint32_t>(instances.size());

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
	accel_options.motionOptions.numKeys = 1;

	uint64_t* compacted_size_buffer;
	CCE(cudaMalloc(reinterpret_cast<void**>(&compacted_size_buffer), sizeof(uint64_t)));

	OptixAccelEmitDesc emit_desc;
	emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emit_desc.result = reinterpret_cast<CUdeviceptr>(compacted_size_buffer);

    OptixAccelBufferSizes ias_buffer_sizes;
    COE(optixAccelComputeMemoryUsage(
		context_, 
		&accel_options, 
		&build_input, 
		1, 
		&ias_buffer_sizes));

    void* temp_buffer;
	CCE(cudaMalloc(&temp_buffer, ias_buffer_sizes.tempSizeInBytes));

	void* output_buffer;
	CCE(cudaMalloc(&output_buffer, ias_buffer_sizes.outputSizeInBytes));

    COE(optixAccelBuild(
		context_,
		nullptr,
		&accel_options,
		&build_input,
		1,
		reinterpret_cast<CUdeviceptr>(temp_buffer),
		ias_buffer_sizes.tempSizeInBytes,
		reinterpret_cast<CUdeviceptr>(output_buffer),
		ias_buffer_sizes.outputSizeInBytes,
		&as_handle,
		&emit_desc, 
		1));
	
	CCE(cudaDeviceSynchronize());
	CCE(cudaGetLastError());

    uint64_t compacted_size;
	CCE(cudaMemcpy(&compacted_size, compacted_size_buffer, sizeof(uint64_t), cudaMemcpyDeviceToHost));

	CCE(cudaMalloc(&ias_buffer_, compacted_size));
	COE(optixAccelCompact(
		context_,
		nullptr,
		as_handle,
		reinterpret_cast<CUdeviceptr>(ias_buffer_),
		compacted_size,
		&as_handle));

	CCE(cudaDeviceSynchronize());
	CCE(cudaGetLastError());

	CCE(cudaFree(output_buffer));
	CCE(cudaFree(temp_buffer));
	CCE(cudaFree(compacted_size_buffer));
	CCE(cudaFree(instance_buffer));

	return as_handle;
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

	std::vector<SbtRecord<HitGroupData>> hitgroup_records;
	for (const auto& object : world_info_->objects_)
	{
		SbtRecord<HitGroupData> rec{};
		COE(optixSbtRecordPackHeader(hit_programs_[enum_cast(object.type) - 1], &rec));
		rec.data.texture = world_info_->textures_[object.texture_id];
		rec.data.material = world_info_->materials_[object.material_id];
		rec.data.object = object;
		hitgroup_records.push_back(rec);
	}

	CCE(cudaMalloc(reinterpret_cast<void**>(&d_hit_records_), hitgroup_records.size() * sizeof(SbtRecord<HitGroupData>)));
	CCE(cudaMemcpy(d_hit_records_, hitgroup_records.data(), hitgroup_records.size() * sizeof(SbtRecord<HitGroupData>), cudaMemcpyHostToDevice));

	sbt_.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(d_hit_records_);
	sbt_.hitgroupRecordStrideInBytes = sizeof(SbtRecord<HitGroupData>);
	sbt_.hitgroupRecordCount = static_cast<uint32_t>(hitgroup_records.size());
}
