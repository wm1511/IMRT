#pragma once
#include "Ray.cuh"
#include "Transform.cuh"

class Camera
{
public:
	__host__ __device__ Camera(const float2 rotation, const float3 position, const float res_x, const float res_y, const float lens_radius, const float focal_distance, const float fov)
        : camera_to_screen_(perspective(fov, 1e-2f, 1000.f))
	{
		camera_to_world_ = rotate_x(rotation.x) * rotate_y(rotation.y) * translate(position);
		lens_radius_ = lens_radius;
        focal_distance_ = focal_distance;

        const float aspect_ratio = res_x / res_y;
        const float2 min = aspect_ratio > 1.0f ? make_float2(-aspect_ratio, -1.0f) : make_float2(-1.0f, -1.0f / aspect_ratio);
        const float2 max = aspect_ratio > 1.0f ? make_float2(aspect_ratio, 1.0f) : make_float2(1.0f, 1.0f / aspect_ratio);

        screen_to_raster_ =
            scale(make_float3(res_x, res_y, 1.0f)) *
            scale(1.0f / make_float3(max.x - min.x, min.y - max.y, 1.0f)) *
            translate(make_float3(-min.x, max.y, 0.0f));
        raster_to_screen_ = invert(screen_to_raster_);
        raster_to_camera_ = invert(camera_to_screen_) * raster_to_screen_;
    }

	__host__ __device__ void update(const float2 rotation, const float3 position, const float lens_radius, const float focal_distance, const float fov)
	{
		camera_to_screen_ = perspective(fov, 1e-2f, 1000.f);
		camera_to_world_ = rotate_x(rotation.x) * rotate_y(rotation.y) * translate(position);
		lens_radius_ = lens_radius;
        focal_distance_ = focal_distance;
	}

    __host__ __device__ Ray cast_ray(const uint32_t pixel_x, const uint32_t pixel_y, uint32_t* random_state) const
	{
		const float3 p_screen = make_float3((float)pixel_x, (float)pixel_y, 0.0f);
		const float3 p_camera = raster_to_camera_.transform(p_screen);
		auto ray = Ray(make_float3(0.0f), normalize(p_camera));

	    if (lens_radius_ > 0.0f) 
		{
			const float ft = focal_distance_ / ray.direction_.z;
			const float3 p_focus = ray.position(ft);

	        ray.origin_ = lens_radius_ * disk_random(random_state);
	        ray.direction_ = normalize(p_focus - ray.origin_);
	    }
	    return camera_to_world_.transform(ray);
	}



protected:
    Transform camera_to_world_;
    Transform camera_to_screen_, raster_to_camera_;
    Transform screen_to_raster_, raster_to_screen_;
    float lens_radius_, focal_distance_;
};

//class Camera
//{
//public:
//	__host__ __device__ Camera(const float3 camera_position, const float3 camera_direction, const float fov, const float aspect_ratio, const float aperture, const float focus_distance) : aspect_ratio_(aspect_ratio)
//	{
//		update(camera_position, camera_direction, fov, aperture, focus_distance);
//	}
//
//	__host__ __device__ void update(const float3 camera_position, const float3 camera_direction, const float fov, const float aperture, const float focus_distance)
//	{
//		const float3 up = make_float3(0.0f, -1.0f, 0.0f);
//		const float viewport_height = 2.0f * tan(fov * 0.5f);
//		const float viewport_width = viewport_height * aspect_ratio_;
//		const float3 target = normalize(camera_direction);
//
//		origin_ = camera_position;
//		lens_radius_ = aperture * 0.5f;
//
//		u_ = versor(cross(up, target));
//		v_ = cross(target, u_);
//		horizontal_map_ = focus_distance * viewport_width * u_;
//		vertical_map_ = focus_distance * viewport_height * v_;
//		starting_point_ = origin_ - horizontal_map_ * 0.5f - vertical_map_ * 0.5f - focus_distance * target;
//	}
//
//	__host__ __device__ Ray cast_ray(uint32_t* random_state, const float screen_x, const float screen_y) const
//	{
//		const float3 random_on_lens = lens_radius_ * disk_random(random_state);
//		const float3 ray_offset = u_ * random_on_lens.x + v_ * random_on_lens.y;
//		return {origin_ + ray_offset, starting_point_ + screen_x * horizontal_map_ + screen_y * vertical_map_ - origin_ - ray_offset};
//	}
//
//private:
//	float3 origin_{};
//	float3 starting_point_{};
//	float3 horizontal_map_{}, vertical_map_{};
//	float3 u_{}, v_{};
//	float lens_radius_{};
//	float aspect_ratio_;
//};