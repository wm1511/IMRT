#include "Renderer.hpp"

#include "../imgui/imgui.h"

#include <chrono>

namespace imrt
{
	void Renderer::draw()
	{
		ImGui::Begin("Control panel");
		if (ImGui::Button("Render", {ImGui::GetContentRegionAvail().x, 0}))
		{
			render();
		}
		ImGui::End();

		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0.0f, 0.0f});
		ImGui::Begin("Viewport");
		width = static_cast<uint32_t>(ImGui::GetContentRegionAvail().x);
		height = static_cast<uint32_t>(ImGui::GetContentRegionAvail().y);

		if (image)
			ImGui::Image(reinterpret_cast<ImU64>(image->getDescriptorSet()),
			             {static_cast<float>(image->getWidth()), static_cast<float>(image->getHeight())});
		ImGui::End();
		ImGui::PopStyleVar();
	}

	void Renderer::render()
	{
		if (!image || width != image->getWidth() || height != image->getHeight())
		{
			image = std::make_unique<Image>(width, height);
			delete[] imageData;
			imageData = new uint32_t[static_cast<uint64_t>(width * height)];
		}

		image->setData(imageData);
	}
}