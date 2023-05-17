// ReSharper disable once CppPrecompiledHeaderIsNotIncluded
#include "OptixPrograms.cuh"
#include "LaunchParams.hpp"

#include "../common/Math.cuh"
#include "../common/Color.cuh"

#include <optix_device.h>

__constant__ LaunchParams launch_params;

static __forceinline__ __device__ void trace(const float3 origin, const float3 direction, uint32_t& u0, uint32_t& u1)
{
    optixTrace(
        launch_params.traversable,
		origin,
		direction,
		kTMin,
		FLT_MAX,
		0.0f,
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		0,
		1,
		0,
        u0, u1);
}

static __forceinline__ __device__ void* unpack_pointer(const uint32_t i0, const uint32_t i1)
{
	const uint64_t u_ptr = (uint64_t)i0 << 32 | i1;
	const auto ptr = (void*)u_ptr;
	return ptr;
}

static __forceinline__ __device__ void pack_pointer(void* ptr, uint32_t& i0, uint32_t& i1)
{
	const auto u_ptr = (uint64_t)ptr;
	i0 = u_ptr >> 32;
	i1 = u_ptr & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T* get_prd()
{
	const uint32_t u0 = optixGetPayload_0();
	const uint32_t u1 = optixGetPayload_1();
	return (T*)unpack_pointer(u0, u1);
}

extern "C" __global__ void __closesthit__radiance()
{
	const HitGroupData* sbt_data = (HitGroupData*)optixGetSbtDataPointer();

	const uint32_t prim_id = optixGetPrimitiveIndex();
	const uint3 index = sbt_data->index[prim_id];
	const float3& a = sbt_data->vertex[index.x];
	const float3& b = sbt_data->vertex[index.y];
	const float3& c = sbt_data->vertex[index.z];
	const float3 normal = normalize(cross(b - a, c - a));

	const float3 ray_dir = optixGetWorldRayDirection();
	const float cos_dn = 0.2f + 0.8f * fabsf(dot(ray_dir, normal));
	float3& prd = *get_prd<float3>();
	prd = cos_dn * sbt_data->color;
}

extern "C" __global__ void __anyhit__radiance()
{

}

extern "C" __global__ void __miss__radiance()
{
	float3& prd = *get_prd<float3>();

	if (launch_params.sky_info.d_hdr_data)
		prd = launch_params.sky_info.hdr_exposure * sample_hdr(optixGetWorldRayDirection(), launch_params.sky_info);
	else
		prd = sample_sky(optixGetWorldRayDirection(), launch_params.sky_info);
}

extern "C" __global__ void __raygen__render_progressive()
{
	const uint3 index = optixGetLaunchIndex();
	const uint32_t pixel_index = index.x + index.y * launch_params.width;

	float3 pixel_color_prd{};

	uint32_t u0, u1;
	pack_pointer(&pixel_color_prd, u0, u1);

	uint32_t random_state = xoshiro(&launch_params.xoshiro_state[pixel_index]);
	const float u = ((float)index.x + pcg(&random_state)) / (float)launch_params.width;
	const float v = ((float)index.y + pcg(&random_state)) / (float)launch_params.height;

	const Ray ray = cast_ray(&random_state, u, v, launch_params.camera_info);

	trace(ray.origin_, ray.direction_, u0, u1);

	launch_params.accumulation_buffer[pixel_index] += make_float4(sqrt(pixel_color_prd), 1.0f);
	launch_params.frame_buffer[pixel_index] = launch_params.accumulation_buffer[pixel_index] / (float)launch_params.sampling_denominator;
}

extern "C" __global__ void __raygen__render_static()
{
	const uint3 index = optixGetLaunchIndex();
	const uint32_t pixel_index = index.x + index.y * launch_params.width;

	float3 pixel_color_prd{};

	uint32_t u0, u1;
	pack_pointer(&pixel_color_prd, u0, u1);

	for (uint32_t i = 0; i < launch_params.sampling_denominator; i++)
	{
		uint32_t random_state = xoshiro(&launch_params.xoshiro_state[pixel_index]);
		const float u = ((float)index.x + pcg(&random_state)) / (float)launch_params.width;
		const float v = ((float)index.y + pcg(&random_state)) / (float)launch_params.height;

		const Ray ray = cast_ray(&random_state, u, v, launch_params.camera_info);

		trace(ray.origin_, ray.direction_, u0, u1);

		launch_params.accumulation_buffer[pixel_index] += make_float4(pixel_color_prd, 1.0f);
	}

	launch_params.frame_buffer[pixel_index] = sqrt(launch_params.accumulation_buffer[pixel_index] / (float)launch_params.sampling_denominator);
}