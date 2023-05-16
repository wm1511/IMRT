#pragma once
#include "../info/CameraInfo.hpp"
#include "../info/SkyInfo.hpp"

#include <optix_types.h>
#include <vector_types.h>

#include <cstdint>

//struct TriangleMeshSbtData
//{
//    float3  color;
//    float3 *vertex;
//    uint3 *index;
//};

struct LaunchParams
{
	uint32_t width{}, height{}, frames_since_refresh{};
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
    //int32_t object_id;
    float3  color;
	float3 *vertex;
	uint3 *index;
};
