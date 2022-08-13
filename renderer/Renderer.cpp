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

#include "Renderer.hpp"
#include "Utils.hpp"

#include "../imgui/imgui.h"
#include <glm/gtc/random.hpp>

#include <thread>
#include <chrono>

void Renderer::draw()
{
	ImGui::Begin("Control panel");
	if (ImGui::Button("Render", {ImGui::GetContentRegionAvail().x, 0}))
	{
		render();
	}
	ImGui::Text("Render time: %.3fs", mRenderTime);
	ImGui::End();

	ImGui::Begin("Viewport");
	mWidth = static_cast<uint32_t>(ImGui::GetContentRegionAvail().x);
	mHeight = static_cast<uint32_t>(ImGui::GetContentRegionAvail().y);

	if (mImage)
		ImGui::Image(reinterpret_cast<ImU64>(mImage->getDescriptorSet()),
			 {static_cast<float>(mImage->getWidth()), static_cast<float>(mImage->getHeight())},
					ImVec2(1, 0), ImVec2(0, 1));
	ImGui::End();
}

glm::vec3 Renderer::trace(Ray& ray, const Scene& scene, int32_t depth)
{
	glm::vec3 color{};
	float rrFactor = 1.0f;
	if (depth >= 5)
	{
		constexpr float rrStopProbability = 0.1f;
		if (glm::linearRand(0.0f, 1.0f) <= rrStopProbability)
			return glm::vec3{0.0f};
		rrFactor = 1.0f / (1.0f - rrStopProbability);
	}

	const Intersection intersection = scene.intersect(ray);
	if (!intersection)
		return glm::vec3{0.0f};

	const glm::vec3 hitPoint = ray.getOrigin() + ray.getDirection() * intersection.getT();
	const glm::vec3 normal = intersection.getObject()->normal(hitPoint);
	ray.setOrigin(hitPoint);

	const glm::vec3 colorChange = intersection.getObject()->getMaterial()->emit(ray, normal);
	const auto materialEmission = glm::vec3{intersection.getObject()->getMaterial()->getEmission()};
	color += (trace(ray, scene, depth + 1) * colorChange + materialEmission) * rrFactor;
	return color;
}

void Renderer::render()
{
	if (!mImage || mWidth != mImage->getWidth() || mHeight != mImage->getHeight())
	{
		mImage = std::make_unique<Image>(mWidth, mHeight);
		delete[] mImageData;
		mImageData = new uint32_t[static_cast<uint64_t>(mWidth * mHeight)];
	}

	Scene scene = Scene::makeCornellBox();
	scene.rebuildBVH(1);

	const auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic)
	for (int32_t y = 0; y < static_cast<int32_t>(mHeight); y++)
	{
		for (int32_t x = 0; x < static_cast<int32_t>(mWidth); x++)
		{
			glm::vec3 pixelColor{0.0f};
			for (uint32_t k = 0; k < Scene::SAMPLES_PER_PIXEL; k++)
			{
				Ray ray;
				ray.setOrigin(glm::vec3{0.0f});
				glm::vec3 camera = Utils::calculateCameraCoords(x, y, mWidth, mHeight);
				camera.x += glm::linearRand(-1.0f, 1.0f) / static_cast<float>(mWidth);
				camera.y += glm::linearRand(-1.0f, 1.0f) / static_cast<float>(mHeight);
				ray.setDirection(normalize(camera - ray.getOrigin()));
				pixelColor += trace(ray, scene, 0) / static_cast<float>(Scene::SAMPLES_PER_PIXEL);
			}
			mImageData[y * mWidth + x] = Utils::convert(clamp(pixelColor, 0.0f, 255.0f));
		}
	}
	mImage->setData(mImageData);

	const auto duration = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start);
	mRenderTime = duration.count();
}