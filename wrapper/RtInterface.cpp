#include "RtInterface.hpp"
#include "../renderer/Renderer.hpp"
#include "../cpu_renderer/CpuRenderer.hpp"
#include "../cuda_renderer/CudaRenderer.cuh"

#include "../imgui/imgui.h"

#include <chrono>

void RtInterface::draw()
{
	//ImGui::ShowDemoWindow();

	{
		ImGui::Begin("Render settings");

		if (ImGui::Button("CUDA render", {ImGui::GetContentRegionAvail().x / 3, 0}))
		{
			frames_rendered_ = 0;
			is_rendering_ = true;
			renderer_ = std::make_unique<CudaRenderer>(&rt_info_);
		}
		ImGui::SameLine();
		if (ImGui::Button("CPU render", {ImGui::GetContentRegionAvail().x / 2, 0}))
		{
			frames_rendered_ = 0;
			is_rendering_ = true;
			renderer_ = std::make_unique<CpuRenderer>(&rt_info_);
		}
		ImGui::SameLine();
		if (ImGui::Button("Slow render", {ImGui::GetContentRegionAvail().x, 0}))
		{
			frames_rendered_ = 0;
			is_rendering_ = true;
			renderer_ = std::make_unique<Renderer>(&rt_info_);
		}
		if (ImGui::Button("Stop rendering", {ImGui::GetContentRegionAvail().x, 0}))
		{
			is_rendering_ = false;
			renderer_.reset();
		}

		if (is_rendering_)
		{
			if (!image_ || width_ != image_->GetWidth() || height_ != image_->GetHeight())
			{
				image_ = std::make_unique<Image>(width_, height_);
				delete[] image_data_;
				image_data_ = new float[static_cast<uint64_t>(4) * height_ * width_];
			}

			const auto start = std::chrono::high_resolution_clock::now();

			renderer_->render(image_data_, width_, height_);

			const auto duration = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start);
			render_time_ = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

			image_->SetData(image_data_);
			frames_rendered_++;
		}

		ImGui::Text("Last render time: %llums", render_time_);
		ImGui::Text("Frames rendered: %llu", frames_rendered_);

		ImGui::Separator();

		if (ImGui::CollapsingHeader("Quality settings", ImGuiTreeNodeFlags_DefaultOpen))
        {
			if (ImGui::TreeNode("Samples per pixel"))
			{
				ImGui::SliderInt("#spp", &rt_info_.samples_per_pixel, 1, 1024, "%d",
							 ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
				if (ImGui::IsItemActive() || ImGui::IsItemHovered())
					ImGui::SetTooltip("Count of samples generated for each pixel");
				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Recursion depth"))
			{
				ImGui::SliderInt("##MaxDepth", &rt_info_.max_depth, 1, INT8_MAX, "%d",
							 ImGuiSliderFlags_AlwaysClamp);
				if (ImGui::IsItemActive() || ImGui::IsItemHovered())
					ImGui::SetTooltip("Maximum depth, that recursion can achieve before being stopped");
				ImGui::TreePop();
			}
		}

		camera_changed_ = edit_camera();
		
		ImGui::End();
	}

	{
		ImGui::Begin("Scene settings");

		/*if (ImGui::CollapsingHeader("Scene settings", ImGuiTreeNodeFlags_DefaultOpen))
		{

		}*/

		ImGui::End();
	}

	{
		ImGui::Begin("Viewport");
		width_ = static_cast<uint32_t>(ImGui::GetContentRegionAvail().x);
		height_ = static_cast<uint32_t>(ImGui::GetContentRegionAvail().y);

		if (image_) ImGui::Image(
				reinterpret_cast<ImU64>(image_->GetDescriptorSet()),
				{static_cast<float>(image_->GetWidth()), static_cast<float>(image_->GetHeight())},
				ImVec2(1, 0), ImVec2(0, 1));
		ImGui::End();
	}
}

bool RtInterface::edit_camera()
{
	bool is_changed = false;
	if (ImGui::CollapsingHeader("Camera settings", ImGuiTreeNodeFlags_DefaultOpen))
    {
	    if (ImGui::TreeNode("Camera position"))
		{
			is_changed |= ImGui::SliderFloat3("##LookOrigin", rt_info_.look_origin, -UINT8_MAX, UINT8_MAX, "%.3f", ImGuiSliderFlags_Logarithmic);
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("Camera target point"))
		{
			is_changed |= ImGui::SliderFloat3("##LookTarget", rt_info_.look_target, -UINT8_MAX, UINT8_MAX, "%.3f", ImGuiSliderFlags_Logarithmic);
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("Vertical field of view"))
		{
			is_changed |= ImGui::SliderAngle("degrees", &rt_info_.fov, 0.0f, 180.0f, "%.3f");
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("Aperture"))
		{
			is_changed |= ImGui::SliderFloat("##Aperture", &rt_info_.aperture, 0, UINT8_MAX, "%.3f", ImGuiSliderFlags_Logarithmic);
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("Focus distance"))
		{
			is_changed |= ImGui::SliderFloat("##FocusDist", &rt_info_.focus_distance, 0, UINT8_MAX, "%.3f", ImGuiSliderFlags_Logarithmic);
			ImGui::TreePop();
		}
	}
	return is_changed;
}

bool RtInterface::edit_scene()
{
	bool is_changed = false;
	return is_changed;
}
