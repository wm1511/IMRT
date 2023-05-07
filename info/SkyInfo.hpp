#pragma once
extern "C"
{
#include "../sky/ArHosekSkyModel.h"
}

#include "stb_image.h"

#include <cstdint>

struct SkyInfo
{
	bool check_hdr(const char* filename)
	{
		return stbi_info(filename, &hdr_width, &hdr_height, &hdr_components);
	}

	void load_hdr(const char* filename)
	{
		stbi_image_free(buffered_hdr_data);
		buffered_hdr_data = stbi_loadf(filename, &hdr_width, &hdr_height, &hdr_components, 3);
	}

	void clear_hdr()
	{
		stbi_image_free(buffered_hdr_data);
		buffered_hdr_data = nullptr;
	}

	void create_sky(const float turbidity = 2.5f, const float albedo_x = 0.5f, const float albedo_y = 0.5f, const float albedo_z = 0.5f, const float elevation = 1.25f)
	{
		ArHosekSkyModelState* local_state_x = arhosek_rgb_skymodelstate_alloc_init(turbidity, albedo_x, elevation);
		ArHosekSkyModelState* local_state_y = arhosek_rgb_skymodelstate_alloc_init(turbidity, albedo_y, elevation);
		ArHosekSkyModelState* local_state_z = arhosek_rgb_skymodelstate_alloc_init(turbidity, albedo_z, elevation);

		for (int32_t i = 0; i < 9; i++)
		{
			sky_config[0][i] = static_cast<float>(local_state_x->configs[0][i]);
			sky_config[1][i] = static_cast<float>(local_state_y->configs[1][i]);
			sky_config[2][i] = static_cast<float>(local_state_z->configs[2][i]);
		}

		sun_radiance.arr[0] = static_cast<float>(local_state_x->radiances[0]);
		sun_radiance.arr[1] = static_cast<float>(local_state_y->radiances[1]);
		sun_radiance.arr[2] = static_cast<float>(local_state_z->radiances[2]);

		arhosekskymodelstate_free(local_state_x);
		arhosekskymodelstate_free(local_state_y);
		arhosekskymodelstate_free(local_state_z);
		
		sun_elevation = elevation;
	}

	float sky_config[3][9]{};
	Float3 sun_radiance{};
	float sun_elevation{};

	float* buffered_hdr_data = nullptr;
	mutable float3* usable_hdr_data = nullptr;
	float hdr_exposure{2.0f};
	int32_t hdr_width{0}, hdr_height{0}, hdr_components{0};
};
