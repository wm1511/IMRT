#pragma once
#include "Camera.hpp"
#include "Sphere.hpp"
#include "Scene.hpp"
#include "Diffuse.hpp"

#include "../wrapper/RtInfo.hpp"
#include "../abstract/IRenderer.hpp"

class Renderer final : public IRenderer
{
public:
	explicit Renderer(const RtInfo* rt_info);
	void render(float* image_data, uint32_t width, uint32_t height) override;

private:
	glm::dvec3 Trace(Ray& ray, const Scene& scene, int32_t depth);

	const RtInfo* rt_info_;
	Scene scene_;
};