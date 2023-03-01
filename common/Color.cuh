#pragma once
#include "../info/SkyInfo.hpp"
#include "World.cuh"
#include "Material.cuh"

#include <cfloat>

__host__ __device__ inline float3 sample_hdr(const Ray& ray, const SkyInfo& sky_info)
{
	const float3 ray_direction = normalize(ray.direction());
    const float longitude = atan2(ray_direction.z, ray_direction.x);
    const float latitude = acos(ray_direction.y);

    const float u = longitude * kInv2Pi;
    const float v = latitude * kInvPi;

    const auto x = (int32_t)(u * (float)sky_info.hdr_width);
    const auto y = (int32_t)(v * (float)sky_info.hdr_height);

    const int32_t hdr_texel_index = x + y * sky_info.hdr_width;

    if (hdr_texel_index < 0)
        return make_float3(0.0f);

    return clamp(sky_info.usable_hdr_data[hdr_texel_index], 0.0f, 1.0f);
}

__host__ __device__ inline float3 sample_sky(const Ray& ray, const SkyInfo& sky_info)
{
	const float3 direction = normalize(ray.direction());
	const float3 sun_direction = make_float3(0.0f, cos(sky_info.sky_state.e), -sin(sky_info.sky_state.e));

	const float gamma = acos(dot(direction, sun_direction));
    const float theta = acos(dot(direction, make_float3(0.0f, 1.0f, 0.0f)));

	const Float9 configs[3]{sky_info.sky_state.c0, sky_info.sky_state.c1, sky_info.sky_state.c2};
    Float3 result{};

    for (int32_t i = 0; i < 3; i++)
    {
	    const float9 config = configs[i].str;

    	const float exp_m = exp(config.f4 * gamma);
    	const float ray_m = cos(gamma) * cos(gamma);
    	const float mie_m = (1.0f + cos(gamma) * cos(gamma)) / pow(1.0f + config.f8 * config.f8 - 2.0f * config.f8 * cos(gamma), 1.5f);
    	const float zenith = sqrt(cos(theta));

        const float sample = (1.0f + config.f0 * exp(config.f1 / (cos(theta) + 0.01f))) * (config.f2 + config.f3 * exp_m + config.f5 * ray_m + config.f6 * mie_m + config.f7 * zenith);
    	result.arr[i] = sample * sky_info.sky_state.r.arr[i];
    }

    result.str *= 50.0f;

    return result.str * 0.0009765625f;
}

__host__ __device__ inline float3 calculate_color(const Ray& ray, World** world, const SkyInfo sky_info, const int32_t max_depth, uint32_t* random_state)
{
	Ray current_ray = ray;
    float3 current_absorption = make_float3(1.0f);

    for (int32_t i = 0; i < max_depth; i++)
    {
	    Intersection intersection{};
        if ((*world)->intersect(current_ray, 0.001f, FLT_MAX, intersection, random_state))
        {
            float3 absorption;
            if (intersection.material->scatter(current_ray, intersection, absorption, random_state))
	            current_absorption *= absorption;
            else 
                return make_float3(0.0f);
        }
        else
        {
            if (sky_info.usable_hdr_data)
            {
                const float3 hdr_color = sample_hdr(current_ray, sky_info);
				return current_absorption * sky_info.hdr_exposure * hdr_color; 
            }
            
	        const float3 sky_color = sample_sky(current_ray, sky_info);
        	return current_absorption * sky_color; 

            /*const float t = 0.5f * (versor(current_ray.direction()).y + 1.0f);
			const float3 color = (1.0f - t) * make_float3(1.0f) + t * make_float3(0.5f, 0.7f, 1.0f);
			return current_absorption * color;*/
        }
    }
	return make_float3(0.0f);
}