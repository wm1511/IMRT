#pragma once
#include <vector_functions.h>

extern "C" __global__ void __intersection__cylinder();
extern "C" __global__ void __closesthit__sphere();
extern "C" __global__ void __closesthit__cylinder();
extern "C" __global__ void __closesthit__triangle();
extern "C" __global__ void __miss__radiance();
extern "C" __global__ void __raygen__render();