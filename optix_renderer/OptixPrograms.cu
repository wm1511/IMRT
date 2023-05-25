// ReSharper disable once CppPrecompiledHeaderIsNotIncluded
#include "OptixPrograms.cuh"
#include "LaunchParams.hpp"

#include "../common/Math.hpp"
#include "../common/Color.hpp"

#include <optix_device.h>

__constant__ LaunchParams launch_params;

static __forceinline__ __device__ void trace(const float3 origin, const float3 direction, uint32_t& u0, uint32_t& u1)
{
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
		0,
        u0, u1);
}

static __forceinline__ __device__ void* unpack_pointer(const uint32_t i0, const uint32_t i1)
{
	const uint64_t u_ptr = (uint64_t)i0 << 32 | i1;
	const auto ptr = (void*)u_ptr;
	return ptr;
}

static __forceinline__ __device__ void pack_pointer(void* ptr, uint32_t& i0, uint32_t& i1)
{
	const auto u_ptr = (uint64_t)ptr;
	i0 = u_ptr >> 32;
	i1 = u_ptr & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T* get_prd()
{
	const uint32_t u0 = optixGetPayload_0();
	const uint32_t u1 = optixGetPayload_1();
	return (T*)unpack_pointer(u0, u1);
}

template<typename T>
static __forceinline__ __device__ T* get_ard()
{ 
	const uint32_t u0 = optixGetAttribute_0();
	const uint32_t u1 = optixGetAttribute_1();
	return (T*)unpack_pointer(u0, u1);
}

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
			float3 normal = normalize(origin + t * direction - cylinder.extreme_b - axis * m);

			uint32_t a0, a1;
			pack_pointer(&normal, a0, a1);

			optixReportIntersection(t, 0, a0, a1);
		}

		const float aa = dot(origin - cylinder.extreme_a, axis);
		const float t_top = -aa / da;
		const float3 top_point = origin + t_top * direction;
		if (length(cylinder.extreme_a - top_point) < cylinder.radius && -da > 0.0f)
		{
			uint32_t a0, a1;
			pack_pointer(&axis, a0, a1);

			optixReportIntersection(t, 0, a0, a1);
		}

		const float t_bottom = -ba / da;
		const float3 bottom_point = origin + t_bottom * direction;
		if (length(cylinder.extreme_b - bottom_point) < cylinder.radius && da > 0.0f)
		{
			axis = -axis;

			uint32_t a0, a1;
			pack_pointer(&axis, a0, a1);

			optixReportIntersection(t, 0, a0, a1);
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

	float3& prd = *get_prd<float3>();
	prd = sbt_data->texture.color(uv);
}

extern "C" __global__ void __closesthit__cylinder()
{
	const HitGroupData* sbt_data = (HitGroupData*)optixGetSbtDataPointer();
	const Cylinder& cylinder = sbt_data->object.cylinder;

	const float3 position = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
	const float3& ard = *get_ard<float3>();
	const float2 uv = make_float2(acosf(normalize(ard).x) / kPi, position.y / (cylinder.extreme_b.y - cylinder.extreme_a.y));

	float3& prd = *get_prd<float3>();
	prd = sbt_data->texture.color(uv);
}

extern "C" __global__ void __closesthit__triangle()
{
	const HitGroupData* sbt_data = (HitGroupData*)optixGetSbtDataPointer();
	const uint32_t prim_idx = optixGetPrimitiveIndex();
	const Model& model = sbt_data->object.model;

	float2 uv = optixGetTriangleBarycentrics();
	//float3 normal;

	if (model.d_uv)
		uv = (1.0f - uv.x - uv.y) * model.d_uv[3ull * prim_idx] +
			uv.x * model.d_uv[3ull * prim_idx + 1] +
			uv.y * model.d_uv[3ull * prim_idx + 2];

    /* if (model.d_normals)
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

	float3 direction = optixGetWorldRayDirection();
	const float3 origin = optixGetWorldRayOrigin() + optixGetRayTmax() * direction;
	const uint3 index = optixGetLaunchIndex();
	uint32_t random_state = xoshiro(&launch_params.xoshiro_state[index.x + index.y * launch_params.width]);*/

	float3& prd = *get_prd<float3>();

	/*if (sbt_data->material.scatter(direction, normal, &random_state))
    {
		uint32_t u0, u1;
		pack_pointer(&prd, u0, u1);

        trace(origin, direction, u0, u1);
    }
    else
    {
        prd = make_float3(0.0f);
    }*/

	prd *= sbt_data->texture.color(uv);
}

extern "C" __global__ void __miss__radiance()
{
	float3& prd = *get_prd<float3>();

	if (launch_params.sky_info.d_hdr_data)
		prd = launch_params.sky_info.hdr_exposure * sample_hdr(optixGetWorldRayDirection(), launch_params.sky_info);
	else
		prd = sample_sky(optixGetWorldRayDirection(), launch_params.sky_info);
}

extern "C" __global__ void __raygen__render_progressive()
{
	const uint3 index = optixGetLaunchIndex();
	const uint32_t pixel_index = index.x + index.y * launch_params.width;

	float3 pixel_color_prd = make_float3(1.0f);

	uint32_t u0, u1;
	pack_pointer(&pixel_color_prd, u0, u1);

	uint32_t random_state = xoshiro(&launch_params.xoshiro_state[pixel_index]);
	const float u = ((float)index.x + pcg(&random_state)) / (float)launch_params.width;
	const float v = ((float)index.y + pcg(&random_state)) / (float)launch_params.height;

	const Ray ray = cast_ray(&random_state, u, v, launch_params.camera_info);

	trace(ray.origin_, ray.direction_, u0, u1);

	launch_params.accumulation_buffer[pixel_index] += make_float4(sqrt(pixel_color_prd), 1.0f);
	launch_params.frame_buffer[pixel_index] = launch_params.accumulation_buffer[pixel_index] / (float)launch_params.sampling_denominator;
}

extern "C" __global__ void __raygen__render_static()
{
	const uint3 index = optixGetLaunchIndex();
	const uint32_t pixel_index = index.x + index.y * launch_params.width;

	float3 pixel_color_prd{};

	uint32_t u0, u1;
	pack_pointer(&pixel_color_prd, u0, u1);

	for (uint32_t i = 0; i < launch_params.sampling_denominator; i++)
	{
		uint32_t random_state = xoshiro(&launch_params.xoshiro_state[pixel_index]);
		const float u = ((float)index.x + pcg(&random_state)) / (float)launch_params.width;
		const float v = ((float)index.y + pcg(&random_state)) / (float)launch_params.height;

		const Ray ray = cast_ray(&random_state, u, v, launch_params.camera_info);

		trace(ray.origin_, ray.direction_, u0, u1);

		launch_params.accumulation_buffer[pixel_index] += make_float4(pixel_color_prd, 1.0f);
	}

	launch_params.frame_buffer[pixel_index] = sqrt(launch_params.accumulation_buffer[pixel_index] / (float)launch_params.sampling_denominator);
}