// ReSharper disable once CppPrecompiledHeaderIsNotIncluded
#include "OptixPrograms.cuh"
#include "LaunchParams.hpp"

#include "../common/Math.cuh"

#include <optix_device.h>

__constant__ LaunchParams launch_params;

extern "C" __global__ void __closesthit__radiance()
{
	
}

extern "C" __global__ void __anyhit__radiance()
{
	
}

extern "C" __global__ void __miss__radiance()
{
	
}

extern "C" __global__ void __raygen__render()
{
	const uint3 i = optixGetLaunchIndex();

	float t = (float)i.y / (float)launch_params.height;
	const float3 color = (1.0f - t) * make_float3(1.0f) + t * make_float3(0.3f, 0.5f, 1.0f);

	const uint32_t index = i.x + i.y * launch_params.width;
	launch_params.frame_buffer[index] = make_float4(color, 1.0f);
}