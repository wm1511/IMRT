#pragma once
#include <vector_types.h>

struct Material;
struct Texture;

struct Intersection
{
	float3 point;
	float3 normal;
	float2 uv;
	Material* material;
	Texture* texture;
};