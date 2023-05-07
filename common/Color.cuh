#pragma once
#include "../info/SkyInfo.hpp"
#include "World.cuh"
#include "Material.cuh"

#include <device_launch_parameters.h>

__host__ __device__ inline float3 sample_hdr(const Ray& ray, const SkyInfo& sky_info)
{
	const float3 ray_direction = normalize(ray.direction_);
    const float longitude = atan2(ray_direction.z, ray_direction.x);
    const float latitude = acos(ray_direction.y);

    const float u = longitude * kInv2Pi;
    const float v = latitude * kInvPi;

    const auto x = (int32_t)(u * (float)sky_info.hdr_width);
    const auto y = (int32_t)(v * (float)sky_info.hdr_height);

    const int32_t hdr_texel_index = x + y * sky_info.hdr_width;

    if (hdr_texel_index < 0 || hdr_texel_index > sky_info.hdr_width * sky_info.hdr_height * 3)
        return make_float3(0.0f);

    return clamp(sky_info.usable_hdr_data[hdr_texel_index], 0.0f, 1.0f);
}

__host__ __device__ inline float3 sample_sky(const Ray& ray, const SkyInfo& sky_info)
{
	const float3 direction = normalize(ray.direction_);
	const float3 sun_direction = make_float3(0.0f, cos(kHalfPi - sky_info.sun_elevation), sin(kHalfPi - sky_info.sun_elevation));

    const float abs_cos_theta = abs(dot(direction, make_float3(0.0f, 1.0f, 0.0f)));
	const float cos_gamma = dot(direction, sun_direction);

    const float ray_m = cos_gamma * cos_gamma;
	const float zenith = sqrt(abs_cos_theta);

    Float3 result{};

    for (int32_t i = 0; i < 3; i++)
    {
	    const auto config = sky_info.sky_config[i];

    	const float exp_m = exp(config[4] * acos(cos_gamma));
    	const float mie_m = (1.0f + cos_gamma * cos_gamma) / pow(1.0f + config[8] * config[8] - 2.0f * config[8] * cos_gamma, 1.5f);
    	const float sample = (1.0f + config[0] * exp(config[1] / (abs_cos_theta + 0.01f))) * (config[2] + config[3] * exp_m + config[5] * ray_m + config[6] * mie_m + config[7] * zenith);

    	result.arr[i] = sample * sky_info.sun_radiance.arr[i];
    }

    return result.str * 0.05f;
}

__host__ __device__ inline float3 calculate_color(const Ray& ray, World** world, const SkyInfo& sky_info, const int32_t max_depth, uint32_t* random_state)
{
	Ray current_ray = ray;
    float3 current_absorption = make_float3(1.0f);

    for (int32_t i = 0; i < max_depth; i++)
    {
	    Intersection intersection{};
        if ((*world)->intersect(current_ray, intersection))
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
				return current_absorption * sky_info.hdr_exposure * sample_hdr(current_ray, sky_info); 

        	return current_absorption * sample_sky(current_ray, sky_info); 
        }
    }
	return make_float3(0.0f);
}