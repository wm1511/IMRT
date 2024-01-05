// Copyright Wiktor Merta 2023
#include "LaunchParams.hpp"

#include "../common/Math.hpp"
#include "../common/Color.hpp"

#include <optix_device.h>

__constant__ LaunchParams launch_params;

extern uint32_t __float_as_uint(float x);
extern float __uint_as_float(uint32_t x);

static __forceinline__ __device__ void trace(float3& origin, float3& direction, float3& color, uint32_t& depth, const uint32_t miss_sbt_index)
{
	uint32_t u0 = __float_as_uint(origin.x);
	uint32_t u1 = __float_as_uint(origin.y);
	uint32_t u2 = __float_as_uint(origin.z);
	uint32_t u3 = __float_as_uint(direction.x);
	uint32_t u4 = __float_as_uint(direction.y);
	uint32_t u5 = __float_as_uint(direction.z);
	uint32_t u6 = __float_as_uint(color.x);
	uint32_t u7 = __float_as_uint(color.y);
	uint32_t u8 = __float_as_uint(color.z);
	uint32_t u9 = depth;

    optixTrace(
        launch_params.traversable,
		origin,
		direction,
		kTMin,
		FLT_MAX,
		0.0f,
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		0,
		1,
		miss_sbt_index,
        u0, u1, u2, u3, u4, u5, u6, u7, u8, u9);

	origin = make_float3(__uint_as_float(u0), __uint_as_float(u1), __uint_as_float(u2));
	direction = make_float3(__uint_as_float(u3), __uint_as_float(u4), __uint_as_float(u5));
	color = make_float3(__uint_as_float(u6), __uint_as_float(u7), __uint_as_float(u8));
	depth = u9;
}

// Custom intersector, the same as in CUDA and CPU renderers
extern "C" __global__ void __intersection__cylinder()
{
	const HitGroupData* sbt_data = (HitGroupData*)optixGetSbtDataPointer();
	const Cylinder& cylinder = sbt_data->object.cylinder;
	const float3 origin = optixGetWorldRayOrigin();
	const float3 direction = optixGetWorldRayDirection();

	const float3 ob = origin - cylinder.extreme_b;
	float3 axis = normalize(cylinder.extreme_a - cylinder.extreme_b);

	const float ba = dot(ob, axis);
	const float da = dot(direction, axis);
	const float od = dot(direction, ob);

	const float a = dot(direction, direction) - da * da;
	const float b = od - da * ba;
	const float c = dot(ob, ob) - ba * ba - cylinder.radius * cylinder.radius;

	const float delta = b * b - a * c;

	if (delta > 0.0f)
	{
		const float sqrt_delta = sqrt(delta);

		const float t1 = (-b - sqrt_delta) / a;
		const float t2 = (-b + sqrt_delta) / a;
		const float t = t1 > t2 ? t2 : t1;

		const float m = da * t + ba;

		if (m > 0.0f && m < length(cylinder.extreme_a - cylinder.extreme_b))
		{
			const float3 position = origin + t * direction;
			const float3 normal = normalize(position- cylinder.extreme_b - axis * m);

			const uint32_t a0 = __float_as_uint(normal.x);
			const uint32_t a1 = __float_as_uint(normal.y);
			const uint32_t a2 = __float_as_uint(normal.z);
			const uint32_t a3 = __float_as_uint(acosf(normalize(normal).x) / kPi);
			const uint32_t a4 = __float_as_uint(position.y / (cylinder.extreme_b.y - cylinder.extreme_a.y));

			optixReportIntersection(t, 0, a0, a1, a2, a3, a4);
		}

		const float aa = dot(origin - cylinder.extreme_a, axis);
		const float t_top = -aa / da;
		const float3 top_point = origin + t_top * direction;
		if (length(cylinder.extreme_a - top_point) < cylinder.radius && -da > 0.0f)
		{
			const uint32_t a0 = __float_as_uint(axis.x);
			const uint32_t a1 = __float_as_uint(axis.y);
			const uint32_t a2 = __float_as_uint(axis.z);
			const uint32_t a3 = __float_as_uint(fracf(top_point.x));
			const uint32_t a4 = __float_as_uint(fracf(top_point.z));

			optixReportIntersection(t, 0, a0, a1, a2, a3, a4);
		}

		const float t_bottom = -ba / da;
		const float3 bottom_point = origin + t_bottom * direction;
		if (length(cylinder.extreme_b - bottom_point) < cylinder.radius && da > 0.0f)
		{
			axis = -axis;

			const uint32_t a0 = __float_as_uint(axis.x);
			const uint32_t a1 = __float_as_uint(axis.y);
			const uint32_t a2 = __float_as_uint(axis.z);
			const uint32_t a3 = __float_as_uint(fracf(bottom_point.x));
			const uint32_t a4 = __float_as_uint(fracf(bottom_point.z));

			optixReportIntersection(t, 0, a0, a1, a2, a3, a4);
		}
	}
}

extern "C" __global__ void __closesthit__sphere()
{
	const HitGroupData* sbt_data = (HitGroupData*)optixGetSbtDataPointer();
	const Sphere& sphere = sbt_data->object.sphere;

	const float3 position = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
	const float3 normal = normalize((position - sphere.center) / sphere.radius);
	const float2 uv = make_float2((atan2(normal.z, normal.x) + kPi) * kInv2Pi, acos(normal.y) * kInvPi);
	float3 direction = make_float3(__uint_as_float(optixGetPayload_3()), __uint_as_float(optixGetPayload_4()), __uint_as_float(optixGetPayload_5()));
	float3 color = make_float3(__uint_as_float(optixGetPayload_6()), __uint_as_float(optixGetPayload_7()), __uint_as_float(optixGetPayload_8()));
	const uint3 index = optixGetLaunchIndex();
	uint32_t random_state = xoshiro(&launch_params.xoshiro_state[index.x + index.y * launch_params.width]);

	if (sbt_data->material.scatter(direction, normal, &random_state))
    {
		color *= sbt_data->texture.color(uv);

		optixSetPayload_0(__float_as_uint(position.x));
		optixSetPayload_1(__float_as_uint(position.y));
		optixSetPayload_2(__float_as_uint(position.z));
		optixSetPayload_3(__float_as_uint(direction.x));
		optixSetPayload_4(__float_as_uint(direction.y));
		optixSetPayload_5(__float_as_uint(direction.z));
		optixSetPayload_9(optixGetPayload_9() - 1);
    }
    else
    {
        optixSetPayload_9(0);
    }

	optixSetPayload_6(__float_as_uint(color.x));
	optixSetPayload_7(__float_as_uint(color.y));
	optixSetPayload_8(__float_as_uint(color.z));
}

extern "C" __global__ void __closesthit__cylinder()
{
	const HitGroupData* sbt_data = (HitGroupData*)optixGetSbtDataPointer();

	const float3 position = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
	const float3 normal = make_float3(__uint_as_float(optixGetAttribute_0()), __uint_as_float(optixGetAttribute_1()), __uint_as_float(optixGetAttribute_2()));
	const float2 uv = make_float2(__uint_as_float(optixGetAttribute_3()), __uint_as_float(optixGetAttribute_4()));
	float3 direction = optixGetWorldRayDirection();
	float3 color = make_float3(__uint_as_float(optixGetPayload_6()), __uint_as_float(optixGetPayload_7()), __uint_as_float(optixGetPayload_8()));
	const uint3 index = optixGetLaunchIndex();
	uint32_t random_state = xoshiro(&launch_params.xoshiro_state[index.x + index.y * launch_params.width]);

	if (sbt_data->material.scatter(direction, normal, &random_state))
    {
		color *= sbt_data->texture.color(uv);

		optixSetPayload_0(__float_as_uint(position.x));
		optixSetPayload_1(__float_as_uint(position.y));
		optixSetPayload_2(__float_as_uint(position.z));
		optixSetPayload_3(__float_as_uint(direction.x));
		optixSetPayload_4(__float_as_uint(direction.y));
		optixSetPayload_5(__float_as_uint(direction.z));
		optixSetPayload_9(optixGetPayload_9() - 1);
    }
    else
    {
        optixSetPayload_9(0);
    }

	optixSetPayload_6(__float_as_uint(color.x));
	optixSetPayload_7(__float_as_uint(color.y));
	optixSetPayload_8(__float_as_uint(color.z));
}

extern "C" __global__ void __closesthit__triangle()
{
	const HitGroupData* sbt_data = (HitGroupData*)optixGetSbtDataPointer();
	const uint32_t prim_idx = optixGetPrimitiveIndex();
	const Model& model = sbt_data->object.model;

	float2 uv = optixGetTriangleBarycentrics();
	float3 normal;

	if (model.d_normals)
    {
	    normal = (1.0f - uv.x - uv.y) * model.d_normals[3ull * prim_idx] + 
			uv.x * model.d_normals[3ull * prim_idx + 1] + 
			uv.y * model.d_normals[3ull * prim_idx + 2];
    }
	else 
	{
		normal = normalize(cross(model.d_vertices[3ull * prim_idx + 1] - model.d_vertices[3ull * prim_idx],
			model.d_vertices[3ull * prim_idx + 2] - model.d_vertices[3ull * prim_idx]));
    }

	if (model.d_uv)
		uv = (1.0f - uv.x - uv.y) * model.d_uv[3ull * prim_idx] +
			uv.x * model.d_uv[3ull * prim_idx + 1] +
			uv.y * model.d_uv[3ull * prim_idx + 2];

	float3 direction = optixGetWorldRayDirection();
	const float3 origin = optixGetWorldRayOrigin() + optixGetRayTmax() * direction;
	float3 color = make_float3(__uint_as_float(optixGetPayload_6()), __uint_as_float(optixGetPayload_7()), __uint_as_float(optixGetPayload_8()));
	const uint3 index = optixGetLaunchIndex();
	uint32_t random_state = xoshiro(&launch_params.xoshiro_state[index.x + index.y * launch_params.width]);

	if (sbt_data->material.scatter(direction, normal, &random_state))
    {
		color *= sbt_data->texture.color(uv);

		optixSetPayload_0(__float_as_uint(origin.x));
		optixSetPayload_1(__float_as_uint(origin.y));
		optixSetPayload_2(__float_as_uint(origin.z));
		optixSetPayload_3(__float_as_uint(direction.x));
		optixSetPayload_4(__float_as_uint(direction.y));
		optixSetPayload_5(__float_as_uint(direction.z));
		optixSetPayload_9(optixGetPayload_9() - 1);
    }
    else
    {
	    optixSetPayload_9(0);
    }

	optixSetPayload_6(__float_as_uint(color.x));
	optixSetPayload_7(__float_as_uint(color.y));
	optixSetPayload_8(__float_as_uint(color.z));
}

extern "C" __global__ void __miss__hdr()
{
	float3 color = make_float3(__uint_as_float(optixGetPayload_6()), __uint_as_float(optixGetPayload_7()), __uint_as_float(optixGetPayload_8()));

	color *= launch_params.sky_info.hdr_exposure * sample_hdr(optixGetWorldRayDirection(), launch_params.sky_info);

	optixSetPayload_6(__float_as_uint(color.x));
	optixSetPayload_7(__float_as_uint(color.y));
	optixSetPayload_8(__float_as_uint(color.z));
	optixSetPayload_9(0);
}

extern "C" __global__ void __miss__sky()
{
	float3 color = make_float3(__uint_as_float(optixGetPayload_6()), __uint_as_float(optixGetPayload_7()), __uint_as_float(optixGetPayload_8()));

	color *= sample_sky(optixGetWorldRayDirection(), launch_params.sky_info);

	optixSetPayload_6(__float_as_uint(color.x));
	optixSetPayload_7(__float_as_uint(color.y));
	optixSetPayload_8(__float_as_uint(color.z));
	optixSetPayload_9(0);
}

// Rendering one pixel
extern "C" __global__ void __raygen__render()
{
	const uint3 index = optixGetLaunchIndex();
	const uint32_t pixel_index = index.x + index.y * launch_params.width;

	float3 pixel_color = make_float3(1.0f);
	float3 origin;
	float3 direction;
	uint32_t depth_remaining = launch_params.depth;

	uint32_t random_state = xoshiro(&launch_params.xoshiro_state[pixel_index]);
	const float u = ((float)index.x + pcg(&random_state)) / (float)launch_params.width;
	const float v = ((float)index.y + pcg(&random_state)) / (float)launch_params.height;

	cast_ray(origin, direction, &random_state, u, v, launch_params.camera_info);
	const uint32_t miss_sbt_index = launch_params.sky_info.d_hdr_data ? 0 : 1;

	while (depth_remaining > 0)
	{
		trace(origin, direction, pixel_color, depth_remaining, miss_sbt_index);
	}

	launch_params.accumulation_buffer[pixel_index] += make_float4(pixel_color, 1.0f);
	launch_params.frame_buffer[pixel_index] = sqrt(launch_params.accumulation_buffer[pixel_index] / (float)launch_params.sampling_denominator);
}