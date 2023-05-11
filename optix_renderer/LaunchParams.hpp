#pragma once
#include <vector_types.h>

#include <cstdint>

struct LaunchParams
{
	uint32_t width{}, height{};
	float4* frame_buffer = nullptr;
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
    int32_t object_id;
};
