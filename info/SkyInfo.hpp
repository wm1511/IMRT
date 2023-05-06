#pragma once
_EXTERN_C
#include "../sky/ArHosekSkyModel.h"
_END_EXTERN_C

#include "stb_image.h"

#include <cstdint>

union SkyConfig
{
	float arr[9]{};

private:
	struct _
	{
		float f0, f1, f2, f3, f4, f5, f6, f7, f8;
	};
};

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
		ArHosekSkyModelState* local_state_x = arhosek_rgb_skymodelstate_alloc_init((float)turbidity, (float)albedo_x, (float)elevation);
		ArHosekSkyModelState* local_state_y = arhosek_rgb_skymodelstate_alloc_init((float)turbidity, (float)albedo_y, (float)elevation);
		ArHosekSkyModelState* local_state_z = arhosek_rgb_skymodelstate_alloc_init((float)turbidity, (float)albedo_z, (float)elevation);

		for (int32_t i = 0; i < 9; i++)
		{
			sky_config_x.arr[i] = static_cast<float>(local_state_x->configs[0][i]);
			sky_config_y.arr[i] = static_cast<float>(local_state_y->configs[1][i]);
			sky_config_z.arr[i] = static_cast<float>(local_state_z->configs[2][i]);
		}

		sun_radiance.arr[0] = static_cast<float>(local_state_x->radiances[0]);
		sun_radiance.arr[1] = static_cast<float>(local_state_y->radiances[1]);
		sun_radiance.arr[2] = static_cast<float>(local_state_z->radiances[2]);

		arhosekskymodelstate_free(local_state_x);
		arhosekskymodelstate_free(local_state_y);
		arhosekskymodelstate_free(local_state_z);
		
		sun_elevation = elevation;
	}

	SkyConfig sky_config_x{};
	SkyConfig sky_config_y{};
	SkyConfig sky_config_z{};
	Float3 sun_radiance{};
	float sun_elevation{};

	float* buffered_hdr_data = nullptr;
	float3* usable_hdr_data = nullptr;
	float hdr_exposure{2.0f};
	int32_t hdr_width{0}, hdr_height{0}, hdr_components{0};
};
