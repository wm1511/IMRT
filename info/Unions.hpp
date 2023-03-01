#pragma once
#include <vector_types.h>

union Float2
{
	float2 str;
	float arr[2];
};

union Float3
{
	float3 str;
	float arr[3];
};

union Float4
{
	float4 str;
	float arr[4];
};

struct float9
{
	float f0, f1, f2, f3, f4, f5, f6, f7, f8;
};

union Float9
{
	float9 str;
	float arr[9];
};