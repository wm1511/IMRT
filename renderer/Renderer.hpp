#pragma once

#include "../wrapper/RtInfo.hpp"
#include "../abstract/IRenderer.hpp"
#include "SceneBuilder.hpp"

#include <functional>

class Renderer final : public IRenderer
{
public:
	explicit Renderer(const RtInfo& rt_info);
	void render(uint32_t* image_data, uint32_t width, uint32_t height) override;

private:
	glm::dvec3 RrTrace(Ray& ray, const Scene& scene, int32_t depth);
	glm::dvec3 DTrace(Ray& ray, const Scene& scene, int32_t depth);

	static uint32_t convert(const glm::vec3 color)
	{
		return 0xff000000 | static_cast<uint8_t>(color.b) << 16 | static_cast<uint8_t>(color.g) << 8 | static_cast<uint8_t>(color.r);
	}

	RtInfo rt_info_;
	std::function<Scene()> prepare_scene_ = &SceneBuilder::MakeCornellBox;
	std::function<glm::dvec3(Renderer*, Ray&, const Scene&, int32_t)> trace_ = &Renderer::DTrace;
};