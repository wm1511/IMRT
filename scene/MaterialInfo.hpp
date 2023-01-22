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
	explicit DiffuseInfo(float albedo[3]) : albedo{albedo[0], albedo[1], albedo[2]} { type = DIFFUSE; }

	float albedo[3];
};

struct SpecularInfo : MaterialInfo
{
	SpecularInfo(float albedo[3], const float fuzziness) : albedo{albedo[0], albedo[1], albedo[2]}, fuzziness(fuzziness) { type = SPECULAR; }

	float albedo[3];
	float fuzziness;
};

struct RefractiveInfo : MaterialInfo
{
	explicit RefractiveInfo(const float refractive_index) : refractive_index(refractive_index) { type = REFRACTIVE; }

	float refractive_index;
};