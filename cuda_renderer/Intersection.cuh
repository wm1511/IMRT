#pragma once
#include <vector_types.h>

class Material;

struct Intersection
{
	float t;
	float3 point;
	float3 normal;
	Material* material;
};