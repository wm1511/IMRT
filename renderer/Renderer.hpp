// Copyright (c) 2022, Wiktor Merta
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "../wrapper/Image.hpp"
#include "SceneBuilder.hpp"

#include <memory>
#include <functional>

class Renderer final : public IDrawable
{
public:
	void draw() override;

private:
	struct RenderInfo
	{
		int32_t sceneIndex{0}, traceType{0}, samplesPerPixel{8}, maxDepth{10}, rrCertainDepth{5};
		glm::dvec3 lookOrigin{0.0, 0.0, 0.0}, lookTarget{0.0, -0.5, -2.5};
		float vfov{1.5f};
		double aperture{0.0}, focusDistance{10.0}, rrStopProbability{0.1};
	};

	void render();
	static uint32_t convert(glm::dvec3 color);
	glm::dvec3 rrTrace(Ray& ray, const Scene& scene, int32_t depth);
	glm::dvec3 dTrace(Ray& ray, const Scene& scene, int32_t depth);

	RenderInfo mRenderInfo;
	std::function<Scene()> mPrepareScene = &SceneBuilder::makeCornellBox;
	std::function<glm::dvec3(Renderer*, Ray&, const Scene&, int32_t)> mTrace = &Renderer::dTrace;
	std::unique_ptr<Image> mImage;
	uint32_t mHeight = 0, mWidth = 0;
	uint32_t* mImageData = nullptr;
	double mRenderTime = 0.0;

};