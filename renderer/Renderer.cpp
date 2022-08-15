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
#include "Camera.hpp"
#include "Utils.hpp"

#include "../imgui/imgui.h"
#include <glm/gtc/random.hpp>

#include <thread>
#include <chrono>
#include <functional>

void Renderer::draw()
{
	{
		ImGui::ShowDemoWindow();

		ImGui::Begin("Control panel");
		if (ImGui::Button("Render", {ImGui::GetContentRegionAvail().x, 0}))
		{
			render();
		}

		ImGui::Dummy({ImGui::GetContentRegionAvail().x / 2 - 4.5f * ImGui::GetFontSize(), 0.0f});
		ImGui::SameLine();
		ImGui::Text("Last render time: %.3fs", mRenderTime);

		ImGui::Separator();

		if (ImGui::CollapsingHeader("Scene settings", ImGuiTreeNodeFlags_DefaultOpen))
		{
			const char* items[] = { "Cornell Box", "Wall" };
			ImGui::Combo("Select render scene", &mRenderInfo.sceneIndex, items, IM_ARRAYSIZE(items));
		}

		if (ImGui::CollapsingHeader("Quality settings", ImGuiTreeNodeFlags_DefaultOpen))
        {
			ImGui::SliderInt("Samples per pixel", &mRenderInfo.samplesPerPixel, 1, 100000, "%d",
							 ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
			if (ImGui::IsItemActive() || ImGui::IsItemHovered())
				ImGui::SetTooltip("CTRL+click to input value manually");
		}

		if (ImGui::CollapsingHeader("Camera settings", ImGuiTreeNodeFlags_DefaultOpen))
        {
			constexpr double min = -DBL_MAX;
			constexpr double zero = 0.0;
			constexpr double max = DBL_MAX;
			if (ImGui::TreeNode("Camera position"))
			{
				ImGui::DragScalar("x", ImGuiDataType_Double, &mRenderInfo.lookOrigin.x, 0.01f, &min, &max, "%.3f");
				ImGui::DragScalar("y", ImGuiDataType_Double, &mRenderInfo.lookOrigin.y, 0.01f, &min, &max, "%.3f");
				ImGui::DragScalar("z", ImGuiDataType_Double, &mRenderInfo.lookOrigin.z, 0.01f, &min, &max, "%.3f");
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("Camera target point"))
			{
				ImGui::DragScalar("x", ImGuiDataType_Double, &mRenderInfo.lookTarget.x, 0.01f, &min, &max, "%.3f");
				ImGui::DragScalar("y", ImGuiDataType_Double, &mRenderInfo.lookTarget.y, 0.01f, &min, &max, "%.3f");
				ImGui::DragScalar("z", ImGuiDataType_Double, &mRenderInfo.lookTarget.z, 0.01f, &min, &max, "%.3f");
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("Vertical field of view"))
			{
				ImGui::SliderAngle("degrees", &mRenderInfo.vfov, 0.0f, 180.0f, "%.3f");
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("Aperture"))
			{
				ImGui::DragScalar("diameter", ImGuiDataType_Double, &mRenderInfo.aperture, 0.001f, &zero, &max, "%.3f");
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("Focus distance"))
			{
				ImGui::DragScalar("distance", ImGuiDataType_Double, &mRenderInfo.focusDistance, 0.05f, &zero, &max, "%.3f");
				ImGui::TreePop();
			}
		}
		
		ImGui::End();
	}

	{
		ImGui::Begin("Viewport");
		mWidth = static_cast<uint32_t>(ImGui::GetContentRegionAvail().x);
		mHeight = static_cast<uint32_t>(ImGui::GetContentRegionAvail().y);

		if (mImage)
			ImGui::Image(reinterpret_cast<ImU64>(mImage->getDescriptorSet()),
				 {static_cast<float>(mImage->getWidth()), static_cast<float>(mImage->getHeight())},
						ImVec2(1, 0), ImVec2(0, 1));
		ImGui::End();
	}
}

Scene Renderer::prepareSelectedScene() const
{
	if (mRenderInfo.sceneIndex == 1)
		return Scene::makeWall();
	return Scene::makeCornellBox();
}

glm::dvec3 Renderer::trace(Ray& ray, const Scene& scene, int32_t depth)
{
	glm::dvec3 color{};
	double rrFactor = 1.0;
	if (depth >= 5)
	{
		constexpr double rrStopProbability = 0.1;
		if (glm::linearRand(0.0, 1.0) <= rrStopProbability)
			return glm::dvec3{0.0};
		rrFactor = 1.0 / (1.0 - rrStopProbability);
	}

	const Intersection intersection = scene.intersect(ray);
	if (!intersection)
		return glm::dvec3{0.0};

	const glm::dvec3 hitPoint = ray.getOrigin() + ray.getDirection() * intersection.t;
	const glm::dvec3 normal = intersection.object->normal(hitPoint);
	ray.setOrigin(hitPoint);

	const glm::dvec3 colorChange = intersection.object->getMaterial()->emit(ray, normal);
	const auto materialEmission = glm::dvec3{intersection.object->getMaterial()->getEmission()};
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

	const Camera camera(mRenderInfo.lookOrigin, 
						mRenderInfo.lookTarget, 
						mRenderInfo.vfov,
	                    static_cast<double>(mWidth) / static_cast<double>(mHeight), 
						mRenderInfo.aperture,
	                    mRenderInfo.focusDistance);

	Scene scene = prepareSelectedScene();
	scene.rebuildBVH(1);

	const auto start = std::chrono::high_resolution_clock::now();

 #pragma omp parallel for schedule(dynamic)
	for (int32_t y = 0; y < static_cast<int32_t>(mHeight); y++)
	{
		for (int32_t x = 0; x < static_cast<int32_t>(mWidth); x++)
		{
			glm::dvec3 pixelColor{0.0};
			for (int32_t k = 0; k < mRenderInfo.samplesPerPixel; k++)
			{
				Ray ray = camera.getRay((x + glm::linearRand(0.0, 1.0)) / mWidth, (y + glm::linearRand(0.0, 1.0)) / mHeight);
				pixelColor += trace(ray, scene, 0) / static_cast<double>(mRenderInfo.samplesPerPixel);
			}
			mImageData[y * mWidth + x] = Utils::convert(clamp(pixelColor, 0.0, 255.0));
		}
	}
	mImage->setData(mImageData);

	const auto duration = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start);
	mRenderTime = duration.count();
}
