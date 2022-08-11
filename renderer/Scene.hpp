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
#include "Intersection.hpp"

#include <vector>

struct SceneInfo
{
	uint32_t samplesPerPixel = 8;
};

class Scene
{
public:
	[[nodiscard]] Intersection intersect(const Ray& ray) const;

	[[nodiscard]] static Scene makeCornellBox();
	[[nodiscard]] SceneInfo getInfo() const;

private:
	Scene() = default;
	void add(std::shared_ptr<Object> object, const std::shared_ptr<Material>& material);

	std::vector<std::shared_ptr<Object>> mObjects;
	SceneInfo mSceneInfo;

};