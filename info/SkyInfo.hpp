#pragma once
#include "Unions.hpp"

_EXTERN_C
#include "../sky/ArHosekSkyModel.h"
_END_EXTERN_C

#include "stb_image.h"

#include <cstdint>

struct SkyState
{
	Float9 c0{};
	Float9 c1{};
	Float9 c2{};
	Float3 r{};
	float e{};
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

	void create_sky(const float turbidity, float albedo[3], const float elevation)
	{
		ArHosekSkyModelState* local_state_x = arhosek_rgb_skymodelstate_alloc_init(turbidity, albedo[0], elevation);
		ArHosekSkyModelState* local_state_y = arhosek_rgb_skymodelstate_alloc_init(turbidity, albedo[1], elevation);
		ArHosekSkyModelState* local_state_z = arhosek_rgb_skymodelstate_alloc_init(turbidity, albedo[2], elevation);

		for (int32_t i = 0; i < 9; i++)
		{
			sky_state.c0.arr[i] = (float)local_state_x->configs[0][i];
			sky_state.c1.arr[i] = (float)local_state_y->configs[1][i];
			sky_state.c2.arr[i] = (float)local_state_z->configs[2][i];
		}

		sky_state.r.arr[0] = (float)local_state_x->radiances[0];
		sky_state.r.arr[1] = (float)local_state_y->radiances[1];
		sky_state.r.arr[2] = (float)local_state_z->radiances[2];

		arhosekskymodelstate_free(local_state_x);
		arhosekskymodelstate_free(local_state_y);
		arhosekskymodelstate_free(local_state_z);
		
		sky_state.e = elevation;
	}

	SkyState sky_state{};

	float* buffered_hdr_data = nullptr;
	float3* usable_hdr_data = nullptr;
	float hdr_exposure{2.0f};
	int32_t hdr_width{0}, hdr_height{0}, hdr_components{0};
};
