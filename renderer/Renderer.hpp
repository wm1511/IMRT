#pragma once
#include "Camera.hpp"
#include "Sphere.hpp"
#include "Scene.hpp"
#include "Diffuse.hpp"

#include "../scene/RtInfo.hpp"
#include "../abstract/IRenderer.hpp"

class Renderer final : public IRenderer
{
public:
	explicit Renderer(const RenderInfo* render_info);
	void render(float* image_data) override;

private:
	glm::dvec3 Trace(Ray& ray, const Scene& scene, int32_t depth);

	const RenderInfo* render_info_;
	Scene scene_;
};