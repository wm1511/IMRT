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
#include "Scene.hpp"

#include <memory>

class Renderer final : public IDrawable
{
public:
	void draw() override;

private:
	void render();
	glm::vec3 trace(Ray& ray, const Scene& scene, int32_t depth);

	std::unique_ptr<Image> mImage;
	uint32_t mHeight = 0, mWidth = 0;
	uint32_t* mImageData = nullptr;
	float mRenderTime = 0.0f;
};