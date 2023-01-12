#pragma once
class Material;

struct Intersection
{
	float t;
	float3 point;
	float3 normal;
	Material* material;
};