#pragma once

#include "../wrapper/RtInfo.hpp"
#include "../abstract/IRenderer.hpp"
#include "SceneBuilder.hpp"

#include <functional>

class Renderer final : public IRenderer
{
public:
	explicit Renderer(const RtInfo* rt_info);
	void render(float* image_data, uint32_t width, uint32_t height) override;

private:
	glm::dvec3 RrTrace(Ray& ray, const Scene& scene, int32_t depth);
	glm::dvec3 DTrace(Ray& ray, const Scene& scene, int32_t depth);

	const RtInfo* rt_info_;
	std::function<Scene()> prepare_scene_ = &SceneBuilder::MakeCornellBox;
	std::function<glm::dvec3(Renderer*, Ray&, const Scene&, int32_t)> trace_ = &Renderer::DTrace;
};