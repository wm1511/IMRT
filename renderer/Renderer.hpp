#pragma once

#include "../wrapper/Image.hpp"

#include <memory>

namespace imrt
{
	class Renderer final : public IDrawable
	{
	public:
		void draw() override;

	private:
		void render();

		std::unique_ptr<Image> image;
		uint32_t height = 0, width = 0;
		uint32_t* imageData = nullptr;
	};
}