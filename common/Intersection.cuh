#pragma once
#include <vector_types.h>

class Material;

struct Intersection
{
	float3 point;
	float3 normal;
	float2 uv;
	Material* material;
	float t;
};