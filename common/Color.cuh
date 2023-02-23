#pragma once
#include "World.cuh"
#include "Material.cuh"

#include <cfloat>

__host__ __device__ inline float3 calculate_color(const Ray& ray, World** world, const float3* hdr_data, const RenderInfo render_info, uint32_t* random_state)
{
	Ray current_ray = ray;
    float3 current_absorption = make_float3(1.0f);
    const int32_t max_depth = render_info.max_depth;

    for (int32_t i = 0; i < max_depth; i++)
    {
	    Intersection intersection{};
        if ((*world)->intersect(current_ray, 0.001f, FLT_MAX, intersection))
        {
            Ray scattered_ray({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f});
            float3 absorption;
            if (intersection.material->scatter(current_ray, intersection, absorption, scattered_ray, random_state))
            {
	            current_absorption *= absorption;
                current_ray = scattered_ray;
            }
            else return make_float3(0.0f);
        }
        else
        {
            if (hdr_data)
            {
                const float3 ray_direction = normalize(current_ray.direction());
                const float longitude = atan2(ray_direction.z, ray_direction.x);
	            const float latitude = acos(ray_direction.y);

			    const float u = longitude * kInv2Pi;
			    const float v = latitude * kInvPi;

	            const auto x = (int32_t)(u * (float)render_info.hdr_width);
	            const auto y = (int32_t)(v * (float)render_info.hdr_height);

	            const int32_t hdr_texel_index = x + y * render_info.hdr_width;
	            const float3 hdr_color = clamp(hdr_data[hdr_texel_index], 0.0f, 1.0f);
				return current_absorption * render_info.hdr_exposure * hdr_color; 
            }

            const float t = 0.5f * (versor(current_ray.direction()).y + 1.0f);
			const float3 color = (1.0f - t) * make_float3(1.0f) + t * make_float3(0.5f, 0.7f, 1.0f);
			return current_absorption * color;
        }
    }
	return make_float3(0.0f);
}