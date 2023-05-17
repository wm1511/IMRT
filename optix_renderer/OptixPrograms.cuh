#pragma once
#include <vector_functions.h>

extern "C" __global__ void __closesthit__radiance();
extern "C" __global__ void __anyhit__radiance();
extern "C" __global__ void __miss__radiance();
extern "C" __global__ void __raygen__render_progressive();
extern "C" __global__ void __raygen__render_static();