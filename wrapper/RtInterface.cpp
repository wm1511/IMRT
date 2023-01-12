#include "RtInterface.hpp"
#include "../renderer/Renderer.hpp"
#include "../cpu_renderer/CpuRenderer.hpp"
#include "../cuda_renderer/CudaRenderer.cuh"

#include "../imgui/imgui.h"

#include <chrono>

void RtInterface::draw()
{
	{
		//ImGui::ShowDemoWindow();

		ImGui::Begin("Control panel");

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
				image_data_ = new float[static_cast<uint64_t>(width_) * height_ * 4];
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

		if (ImGui::CollapsingHeader("Scene settings", ImGuiTreeNodeFlags_DefaultOpen))
		{
			const char* items[] = { "Cornell Box", "Ukraine", "Choinka"};
			ImGui::Combo("##SelectScene", &rt_info_.scene_index, items, IM_ARRAYSIZE(items));
			if (ImGui::IsItemActive() || ImGui::IsItemHovered())
				ImGui::SetTooltip("Select a scene to be rendered");
			if (ImGui::IsItemEdited())
			{
				if (rt_info_.scene_index == 0)
				{
					rt_info_.look_target_x = -1.0f;
					rt_info_.look_target_y = -1.5f;
					rt_info_.look_target_z = -4.5f;
					rt_info_.fov = 1.5f;
					rt_info_.aperture = 0.0f;
					rt_info_.focus_distance = 10.0f;
				}
				else if (rt_info_.scene_index == 1)
				{
					rt_info_.look_target_x = 0.0f;
					rt_info_.look_target_y = -0.5f;
					rt_info_.look_target_z = -2.5f;
					rt_info_.fov = 1.5f;
					rt_info_.aperture = 0.1f;
					rt_info_.focus_distance = 10.0f;
				}
				else if (rt_info_.scene_index == 2)
				{
					rt_info_.look_target_x = 0.0f;
					rt_info_.look_target_y = 0.0f;
					rt_info_.look_target_z = -1.0f;
					rt_info_.fov = 1.5f;
					rt_info_.aperture = 0.0f;
					rt_info_.focus_distance = 10.0f;
				}
			}
		}

		if (ImGui::CollapsingHeader("Quality settings", ImGuiTreeNodeFlags_DefaultOpen))
        {
			ImGui::SliderInt("##spp", &rt_info_.samples_per_pixel, 1, 1024, "%d",
							 ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
			if (ImGui::IsItemActive() || ImGui::IsItemHovered())
				ImGui::SetTooltip("Count of samples generated for each pixel");

			ImGui::Text("Recursion termination method");
			ImGui::SameLine();
			ImGui::RadioButton("Depth", &rt_info_.trace_type, 0);
			ImGui::SameLine();
			ImGui::RadioButton("Russian roulette",&rt_info_.trace_type, 1);

			if (rt_info_.trace_type != 0)
				ImGui::BeginDisabled(true);
			if (ImGui::TreeNode("Depth"))
			{
				ImGui::SliderInt("##MaxDepth", &rt_info_.max_depth, 1, INT8_MAX, "%d",
								 ImGuiSliderFlags_AlwaysClamp);
				if (ImGui::IsItemActive() || ImGui::IsItemHovered())
					ImGui::SetTooltip("Maximum depth, that recursion can achieve before being stopped");
				ImGui::TreePop();
			}
			if (rt_info_.trace_type != 0)
				ImGui::EndDisabled();
			
			if (rt_info_.trace_type != 1)
				ImGui::BeginDisabled(true);
			if (ImGui::TreeNode("Russian roulette"))
			{
				ImGui::SliderInt("##CertDepth", &rt_info_.rr_certain_depth, 1, INT8_MAX, "%d",
								 ImGuiSliderFlags_AlwaysClamp);
				if (ImGui::IsItemActive() || ImGui::IsItemHovered())
					ImGui::SetTooltip("Recursion depth, that will be always achieved before stopping recursion");

				ImGui::SliderFloat("##StopProb", &rt_info_.rr_stop_probability, 0, 1, "%.3f");
				if (ImGui::IsItemActive() || ImGui::IsItemHovered())
					ImGui::SetTooltip("Probability to stop recursion");

				ImGui::TreePop();
			}
			if (rt_info_.trace_type != 1)
				ImGui::EndDisabled();
		
		}

		if (ImGui::CollapsingHeader("Camera settings", ImGuiTreeNodeFlags_DefaultOpen))
        {
			if (ImGui::TreeNode("Camera position"))
			{
				ImGui::SliderFloat("x", &rt_info_.look_origin_x, -UINT8_MAX, UINT8_MAX, "%.3f", ImGuiSliderFlags_Logarithmic);
				ImGui::SliderFloat("y", &rt_info_.look_origin_y, -UINT8_MAX, UINT8_MAX, "%.3f", ImGuiSliderFlags_Logarithmic);
				ImGui::SliderFloat("z", &rt_info_.look_origin_z, -UINT8_MAX, UINT8_MAX, "%.3f", ImGuiSliderFlags_Logarithmic);
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("Camera target point"))
			{
				ImGui::SliderFloat("x", &rt_info_.look_target_x, -UINT8_MAX, UINT8_MAX, "%.3f", ImGuiSliderFlags_Logarithmic);
				ImGui::SliderFloat("y", &rt_info_.look_target_y, -UINT8_MAX, UINT8_MAX, "%.3f", ImGuiSliderFlags_Logarithmic);
				ImGui::SliderFloat("z", &rt_info_.look_target_z, -UINT8_MAX, UINT8_MAX, "%.3f", ImGuiSliderFlags_Logarithmic);
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("Vertical field of view"))
			{
				ImGui::SliderAngle("degrees", &rt_info_.fov, 0.0f, 180.0f, "%.3f");
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("Aperture"))
			{
				ImGui::SliderFloat("##Aperture", &rt_info_.aperture, 0, UINT8_MAX, "%.3f", ImGuiSliderFlags_Logarithmic);
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("Focus distance"))
			{
				ImGui::SliderFloat("##FocusDist", &rt_info_.focus_distance, 0, UINT8_MAX, "%.3f", ImGuiSliderFlags_Logarithmic);
				ImGui::TreePop();
			}
		}
		
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