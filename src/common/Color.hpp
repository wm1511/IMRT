// Copyright Wiktor Merta 2023
#pragma once
#include "../info/SkyInfo.hpp"
#include "../info/CameraInfo.hpp"
#include "World.hpp"

// Mapping screen coordinates to ray, including lens simulation
__host__ __device__ __forceinline__ Ray cast_ray(uint32_t* random_state, const float screen_x, const float screen_y, const CameraInfo& camera_info)
{
	const float2 random_on_lens = camera_info.lens_radius * disk_random(random_state);
	const float3 ray_offset = camera_info.u * random_on_lens.x + camera_info.v * random_on_lens.y;
	return {camera_info.position + ray_offset,
		camera_info.starting_point + screen_x * camera_info.horizontal_map + screen_y * camera_info.vertical_map - camera_info.position - ray_offset};
}

// Same as above, returning by arguments
__host__ __device__ __forceinline__ void cast_ray(float3& origin, float3& direction, uint32_t* random_state, const float screen_x, const float screen_y, const CameraInfo& camera_info)
{
	const float2 random_on_lens = camera_info.lens_radius * disk_random(random_state);
	const float3 ray_offset = camera_info.u * random_on_lens.x + camera_info.v * random_on_lens.y;
	origin = camera_info.position + ray_offset;
	direction = camera_info.starting_point + screen_x * camera_info.horizontal_map + screen_y * camera_info.vertical_map - camera_info.position - ray_offset;
}

// Sampling HDR image using spherical projection
__host__ __device__ __forceinline__ float3 sample_hdr(const float3 direction, const SkyInfo& sky_info)
{
	const float3 ray_direction = normalize(direction);
    const float longitude = atan2(ray_direction.z, ray_direction.x);
    const float latitude = acos(ray_direction.y);

    const float u = longitude * kInv2Pi;
    const float v = latitude * kInvPi;

    const auto x = (int32_t)(u * (float)sky_info.hdr_width);
    const auto y = (int32_t)(v * (float)sky_info.hdr_height);

    const int32_t hdr_texel_index = x + y * sky_info.hdr_width;

    if (hdr_texel_index < 0 || hdr_texel_index > sky_info.hdr_width * sky_info.hdr_height * 3)
        return make_float3(0.0f);

    return clamp(sky_info.d_hdr_data[hdr_texel_index], 0.0f, 1.0f);
}

// Sampling sky model, rewritten from "An Analytic Model for Full Spectral Sky-Dome Radiance" for GPU execution
__host__ __device__ __forceinline__ float3 sample_sky(const float3 direction, const SkyInfo& sky_info)
{
	const float3 ray_direction = normalize(direction);
	const float3 sun_direction = make_float3(0.0f, cos(kHalfPi - sky_info.sun_elevation), sin(kHalfPi - sky_info.sun_elevation));

    const float abs_cos_theta = abs(dot(ray_direction, make_float3(0.0f, 1.0f, 0.0f)));
	const float cos_gamma = dot(ray_direction, sun_direction);

    const float ray_m = cos_gamma * cos_gamma;
	const float zenith = sqrt(abs_cos_theta);

    float result[3];

    for (int32_t i = 0; i < 3; i++)
    {
	    const auto config = sky_info.sky_config[i];

    	const float exp_m = exp(config[4] * acos(cos_gamma));
    	const float mie_m = (1.0f + cos_gamma * cos_gamma) / pow(1.0f + config[8] * config[8] - 2.0f * config[8] * cos_gamma, 1.5f);
    	const float sample = (1.0f + config[0] * exp(config[1] / (abs_cos_theta + 0.01f))) * (config[2] + config[3] * exp_m + config[5] * ray_m + config[6] * mie_m + config[7] * zenith);

    	result[i] = sample;
    }

    return make_float3(result) * sky_info.sun_radiance * 0.05f;
}

// Main path tracing loop used in CPU and CUDA renderers
__host__ __device__ __forceinline__ float3 calculate_color(const Ray& ray, const World* world, const SkyInfo& sky_info, const int32_t max_depth, uint32_t* random_state)
{
	Ray current_ray = ray;
    float3 current_absorption = make_float3(1.0f);

    for (int32_t i = 0; i < max_depth; i++)
    {
	    Intersection intersection{};
        if (world->intersect(current_ray, intersection))
        {
        	if (intersection.material->scatter(current_ray.direction_, intersection.normal, random_state))
        	{
        		current_ray.origin_ = intersection.point;
                current_ray.t_max_ = FLT_MAX;
        	}
            else
            {
	            return current_absorption;
            }

        	current_absorption *= intersection.texture->color(intersection.uv);
        }
        else
        {
            if (sky_info.d_hdr_data)
				return current_absorption * sky_info.hdr_exposure * sample_hdr(current_ray.direction_, sky_info); 

        	return current_absorption * sample_sky(current_ray.direction_, sky_info); 
        }
    }
	return current_absorption;
}