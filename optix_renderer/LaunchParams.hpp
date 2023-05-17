#pragma once
#include "../info/CameraInfo.hpp"
#include "../info/SkyInfo.hpp"

#include <optix_types.h>
#include <vector_types.h>

#include <cstdint>

struct LaunchParams
{
	uint32_t width{}, height{}, sampling_denominator{};
	float4* frame_buffer = nullptr, * accumulation_buffer = nullptr;
    uint4* xoshiro_state = nullptr;
    CameraInfo camera_info{};
    SkyInfo sky_info{};
    OptixTraversableHandle traversable{};
};

struct RayGenData
{
    void *data;
};


struct MissData
{
    void *data;
};


struct HitGroupData
{
    float3  color;
	float3 *vertex;
	uint3 *index;
};
