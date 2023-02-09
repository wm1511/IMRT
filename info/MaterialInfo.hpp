#pragma once

enum MaterialType
{
	UNKNOWN_MATERIAL,
	DIFFUSE,
	SPECULAR,
	REFRACTIVE
};

struct MaterialInfo
{
	MaterialType type{UNKNOWN_MATERIAL};
};

struct DiffuseInfo : MaterialInfo
{
	explicit DiffuseInfo(float3 albedo) : albedo{albedo} { type = DIFFUSE; }

	union
	{
		float3 albedo;
		float albedo_array[3]{};
	};
};

struct SpecularInfo : MaterialInfo
{
	SpecularInfo(float3 albedo, const float fuzziness) : albedo{albedo}, fuzziness(fuzziness) { type = SPECULAR; }

	union
	{
		float3 albedo;
		float albedo_array[3]{};
	};
	float fuzziness;
};

struct RefractiveInfo : MaterialInfo
{
	explicit RefractiveInfo(const float refractive_index) : refractive_index(refractive_index) { type = REFRACTIVE; }

	float refractive_index{};
};