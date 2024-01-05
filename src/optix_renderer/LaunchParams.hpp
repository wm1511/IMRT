// Copyright Wiktor Merta 2023
#pragma once
#include "../common/Object.hpp"
#include "../common/Material.hpp"
#include "../common/Texture.hpp"
#include "../info/CameraInfo.hpp"
#include "../info/SkyInfo.hpp"

#include <optix_types.h>
#include <vector_types.h>

#include <cstdint>

struct LaunchParams
{
	uint32_t width{}, height{}, sampling_denominator{}, depth{};
	float4* frame_buffer = nullptr, * accumulation_buffer = nullptr;
    uint4* xoshiro_state = nullptr;
    CameraInfo camera_info{};
    SkyInfo sky_info{};
    OptixTraversableHandle traversable{};
};

struct RayGenData
{
    void* data;
};


struct MissData
{
    void* data;
};


struct HitGroupData
{
    Object object;
    Material material;
    Texture texture;
};
