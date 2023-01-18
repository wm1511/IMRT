#include "RtInterface.hpp"
//#include "../renderer/Renderer.hpp"
#include "../cpu_renderer/CpuRenderer.hpp"
#include "../cuda_renderer/CudaRenderer.cuh"

#include "../imgui/imgui.h"

#include <chrono>

static void draw_help(const char* text)
{
	ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered())
    {
        ImGui::BeginTooltip();
        ImGui::TextUnformatted(text);
        ImGui::EndTooltip();
    }
}

static void draw_material_list(const std::vector<std::shared_ptr<MaterialInfo>>& material_data, const char* material_types[], uint32_t& selected_material)
{
	ImGui::BeginChild("Materials", {ImGui::GetContentRegionAvail().x, ImGui::GetFontSize() * 6}, true,
		ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_AlwaysAutoResize);
	draw_help("Choose this object's material");
	for (uint32_t n = 0; n < material_data.size(); n++)
	{
		char buffer[16];
		if (sprintf_s(buffer, "%u-%s", n, material_types[material_data[n]->type]) > 0)
		{
			if (ImGui::Selectable(buffer, selected_material == n))
				selected_material = n;
		}
		else
		{
			ImGui::Text("Failed to print material");
		}
	}
	ImGui::EndChild();
}

void RtInterface::init()
{
	float center[3] = {0.0f, 0.0f, -1.0f};
	float green[3] = {0.1f, 0.9f, 0.1f};
	material_data_.push_back(std::make_shared<DiffuseInfo>(green));
	material_data_.push_back(std::make_shared<SpecularInfo>(green, 0.1f));
	material_data_.push_back(std::make_shared<RefractiveInfo>(1.5f));
	object_data_.push_back(std::make_shared<SphereInfo>(center, -0.5f, material_data_[0]));
	object_data_.push_back(std::make_shared<SphereInfo>(center, -0.8f, material_data_[0]));
}

void RtInterface::draw()
{
	ImGui::ShowDemoWindow();

	{
		ImGui::Begin("Render settings");

		if (ImGui::Button("CPU render", {ImGui::GetContentRegionAvail().x / 3, 0}))
		{
			frames_rendered_ = 0;
			is_rendering_ = true;
			renderer_ = std::make_unique<CpuRenderer>(&render_info_);
		}
		ImGui::SameLine();
		if (ImGui::Button("CUDA render", {ImGui::GetContentRegionAvail().x / 2, 0}))
		{
			frames_rendered_ = 0;
			is_rendering_ = true;
			renderer_ = std::make_unique<CudaRenderer>(&render_info_, width_, height_);
		}
		ImGui::SameLine();
		if (ImGui::Button("OPTIX render", {ImGui::GetContentRegionAvail().x, 0}))
		{
			/*frames_rendered_ = 0;
			is_rendering_ = true;
			renderer_ = std::make_unique<Renderer>(&render_info_);*/
		}
		if (ImGui::Button("Stop rendering", {ImGui::GetContentRegionAvail().x, 0}))
		{
			is_rendering_ = false;
		}

		if (is_rendering_)
		{
			if (!image_ || width_ != image_->GetWidth() || height_ != image_->GetHeight())
			{
				image_ = std::make_unique<Image>(width_, height_);
				delete[] image_data_;
				image_data_ = new float[static_cast<uint64_t>(4) * height_ * width_];
				renderer_->recreate_image(width_, height_);
				renderer_->recreate_camera(width_, height_);
			}

			if (camera_changed_)
				renderer_->recreate_camera(width_, height_);

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
				ImGui::SliderInt("##SamplesPerPixel", &render_info_.samples_per_pixel, 1, 1024, "%d",
					ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
				draw_help("Count of samples generated for each pixel");
				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Recursion depth"))
			{
				ImGui::SliderInt("##RecursionDepth", &render_info_.max_depth, 1, INT8_MAX, "%d", ImGuiSliderFlags_AlwaysClamp);
				draw_help("Maximum depth, that recursion can achieve before being stopped");
				ImGui::TreePop();
			}
		}

		camera_changed_ = edit_camera();
		
		ImGui::End();
	}

	{
		ImGui::Begin("Scene settings");

		scene_changed_ = edit_scene();

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
			is_changed |= ImGui::SliderFloat3("##LookOrigin", render_info_.look_origin, -UINT16_MAX, UINT16_MAX, "%.3f",
				ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("Camera target point"))
		{
			is_changed |= ImGui::SliderFloat3("##LookTarget", render_info_.look_target, -UINT16_MAX, UINT16_MAX, "%.3f",
				ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("Vertical field of view"))
		{
			is_changed |= ImGui::SliderAngle("degrees", &render_info_.fov, 0.0f, 180.0f, "%.3f");
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("Aperture"))
		{
			is_changed |= ImGui::SliderFloat("##Aperture", &render_info_.aperture, 0, UINT8_MAX, "%.3f",
				ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("Focus distance"))
		{
			is_changed |= ImGui::SliderFloat("##FocusDist", &render_info_.focus_distance, 0, UINT8_MAX, "%.3f",
				ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
			ImGui::TreePop();
		}
	}
	return is_changed;
}

bool RtInterface::edit_scene()
{
	bool is_changed = false;
	static const char* object_types[]{"Unknown_Object", "Sphere", "Triangle", "Triangle_Mesh"};
	static const char* material_types[]{"Unknown_Material", "Diffuse", "Specular", "Refractive"};
	if (ImGui::CollapsingHeader("Material list"))
	{
		for (uint32_t i = 0; i < material_data_.size(); i++)
		{
			const std::shared_ptr<MaterialInfo> current_material = material_data_[i];
			ImGui::PushID((int32_t)i);
			if (ImGui::TreeNode("Material", "%u-%s", i, material_types[current_material->type]))
			{
				if (ImGui::TreeNode("Properties"))
				{
					ImGui::Text("Material type: %s", material_types[current_material->type]);
					if (current_material->type == UNKNOWN_MATERIAL)
					{
						ImGui::Text("Material properties can't be set for type Unknown_Material");
					}
					else if (current_material->type == DIFFUSE)
					{
						const std::shared_ptr<DiffuseInfo> current_diffuse = std::static_pointer_cast<DiffuseInfo, MaterialInfo>(current_material);
						is_changed |= ImGui::ColorEdit3("Color", current_diffuse->albedo, ImGuiColorEditFlags_Float);
					}
					else if (current_material->type == SPECULAR)
					{
						const std::shared_ptr<SpecularInfo> current_specular = std::static_pointer_cast<SpecularInfo, MaterialInfo>(current_material);
						is_changed |= ImGui::ColorEdit3("Color", current_specular->albedo, ImGuiColorEditFlags_Float);
						is_changed |= ImGui::SliderFloat("Fuzziness", &current_specular->fuzziness, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
						draw_help("Fuzziness of material reflection");
					}
					else if (current_material->type == REFRACTIVE)
					{
						const std::shared_ptr<RefractiveInfo> current_refractive = std::static_pointer_cast<RefractiveInfo, MaterialInfo>(current_material);
						is_changed |= ImGui::SliderFloat("Refractive index", &current_refractive->refractive_index, 0.0f, 4.0f, "%.3f", 
							ImGuiSliderFlags_AlwaysClamp);
						draw_help("Index of refraction between air and current material");
					}
					ImGui::TreePop();
				}
				ImGui::TreePop();
			}
			ImGui::PopID();
		}
	}
	if (ImGui::CollapsingHeader("Object list"))
	{
		for (uint32_t i = 0; i < object_data_.size(); i++)
		{
			const std::shared_ptr<ObjectInfo> current_object = object_data_[i];

			ImGui::PushID((int32_t)i);
			if (ImGui::TreeNode("Object", "%u-%s-%s", i, object_types[current_object->object_type], material_types[current_object->material->type]))
			{
				if (ImGui::TreeNode("Properties"))
				{
					ImGui::Text("Object type: %s", object_types[current_object->object_type]);
					if (current_object->object_type == UNKNOWN_OBJECT)
					{
						ImGui::Text("Object properties can't be set for type Unknown_Object");
					}
					else if (current_object->object_type == SPHERE)
					{
						const std::shared_ptr<SphereInfo> current_sphere = std::static_pointer_cast<SphereInfo, ObjectInfo>(current_object);

						is_changed |= ImGui::SliderFloat3("Center", current_sphere->center, -UINT16_MAX, UINT16_MAX,"%.3f",
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
						is_changed |= ImGui::SliderFloat("Radius", &current_sphere->radius, -UINT16_MAX, UINT16_MAX, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
						draw_help("Radius of sphere can be negative for refractive spheres to make sphere hollow");
					}
					else if (current_object->object_type == TRIANGLE)
					{
						const std::shared_ptr<TriangleInfo> current_triangle = std::static_pointer_cast<TriangleInfo, ObjectInfo>(current_object);

						is_changed |= ImGui::SliderFloat3("Vertex 0", current_triangle->v0, -UINT16_MAX, UINT16_MAX, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
						is_changed |= ImGui::SliderFloat3("Vertex 1", current_triangle->v1, -UINT16_MAX, UINT16_MAX, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
						is_changed |= ImGui::SliderFloat3("Vertex 2", current_triangle->v2, -UINT16_MAX, UINT16_MAX, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
					}
					ImGui::TreePop();
				}

				if (ImGui::TreeNode("Material"))
				{
					static uint32_t selected_material = 0;
		            draw_material_list(material_data_, material_types, selected_material);

					if (ImGui::Button("Set material", {ImGui::GetContentRegionAvail().x, 0}))
					{
						current_object->material = material_data_[selected_material];
					}
					ImGui::TreePop();
				}

				if (ImGui::TreeNode("Transform"))
				{
					is_changed |= ImGui::SliderFloat3("Translation", current_object->transform.translation, -UINT16_MAX, UINT16_MAX, "%.3f", 
							ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_Logarithmic);
					is_changed |= ImGui::SliderFloat3("Scale", current_object->transform.scale, 0.0f, UINT8_MAX, "%.3f", 
							ImGuiSliderFlags_AlwaysClamp);
					is_changed |= ImGui::SliderAngle("Rotation x", &current_object->transform.rotation[0], 0.0f, 360.0f, "%.3f", 
							ImGuiSliderFlags_AlwaysClamp);
					is_changed |= ImGui::SliderAngle("Rotation y", &current_object->transform.rotation[1], 0.0f, 360.0f, "%.3f", 
							ImGuiSliderFlags_AlwaysClamp);
					is_changed |= ImGui::SliderAngle("Rotation z", &current_object->transform.rotation[2], 0.0f, 360.0f, "%.3f", 
							ImGuiSliderFlags_AlwaysClamp);
					ImGui::TreePop();
				}

				if (ImGui::Button("Delete object", {ImGui::GetContentRegionAvail().x, 0}))
				{
					if (object_data_.size() > 1 && object_data_.back() != object_data_[i])
					{
						object_data_.erase(object_data_.begin() + i);
						object_data_[i] = std::move(object_data_.back());
					}
					else
					{
						object_data_.pop_back();
					}
					is_changed = true;
				}

				ImGui::TreePop();
			}
			ImGui::PopID();
		}
	}
	if (ImGui::CollapsingHeader("Add material"))
	{
		static int32_t material_type = UNKNOWN_MATERIAL;

		ImGui::Combo("Material type", &material_type, material_types, IM_ARRAYSIZE(material_types));
		if (material_type == UNKNOWN_MATERIAL)
		{
			ImGui::Text("Material of type Unknown_Material can't be instantiated");
		}
		else if (material_type == DIFFUSE)
		{
			static float new_diffuse_color[3]{0.0f, 0.0f, 0.0f};

			if (ImGui::TreeNode("Properties"))
			{
				ImGui::ColorEdit3("Color", new_diffuse_color, ImGuiColorEditFlags_Float);
				ImGui::TreePop();
			}

			if (ImGui::Button("Create material", {ImGui::GetContentRegionAvail().x, 0}))
			{
				material_data_.push_back(std::make_shared<DiffuseInfo>(new_diffuse_color));
				is_changed = true;
			}
		}
		else if (material_type == SPECULAR)
		{
			static float new_specular_color[3]{0.0f, 0.0f, 0.0f};
			static float new_specular_fuzziness{0.0f};

			if (ImGui::TreeNode("Properties"))
			{
				ImGui::ColorEdit3("Color", new_specular_color, ImGuiColorEditFlags_Float);
				ImGui::SliderFloat("Fuzziness", &new_specular_fuzziness, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
				draw_help("Fuzziness of material reflection");
				ImGui::TreePop();
			}

			if (ImGui::Button("Create material", {ImGui::GetContentRegionAvail().x, 0}))
			{
				material_data_.push_back(std::make_shared<SpecularInfo>(new_specular_color, new_specular_fuzziness));
				is_changed = true;
			}
		}
		else if (material_type == REFRACTIVE)
		{
			static float new_refractive_index_of_refraction{0.0f};

			if (ImGui::TreeNode("Properties"))
			{
				ImGui::SliderFloat("Refractive index", &new_refractive_index_of_refraction, 0.0f, 4.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
				draw_help("Index of refraction between air and current material");
				ImGui::TreePop();
			}

			if (ImGui::Button("Create material", {ImGui::GetContentRegionAvail().x, 0}))
			{
				material_data_.push_back(std::make_shared<RefractiveInfo>(new_refractive_index_of_refraction));
				is_changed = true;
			}
		}
	}
	if (ImGui::CollapsingHeader("Add object"))
	{
		static int32_t object_type = UNKNOWN_OBJECT;
		static uint32_t selected_material = 0;

		ImGui::Combo("Object type", &object_type, object_types, IM_ARRAYSIZE(object_types));
		if (object_type == UNKNOWN_OBJECT)
		{
			ImGui::Text("Object of type Unknown_Object can't be instantiated");
		}
		else if (object_type == SPHERE)
		{
			static float new_sphere_center[3]{0.0f, 0.0f, 0.0f};
			static float new_sphere_radius{1.0f};

			if (ImGui::TreeNode("Properties"))
			{
				ImGui::SliderFloat3("Center", new_sphere_center, -UINT16_MAX, UINT16_MAX,"%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
				ImGui::SliderFloat("Radius", &new_sphere_radius, -UINT16_MAX, UINT16_MAX, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
				draw_help("Radius of sphere can be negative for refractive spheres to make sphere hollow");
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("Material"))
			{
				draw_material_list(material_data_, material_types, selected_material);
				ImGui::TreePop();
			}

			if (ImGui::Button("Create object", {ImGui::GetContentRegionAvail().x, 0}))
			{
				object_data_.push_back(std::make_shared<SphereInfo>(new_sphere_center, new_sphere_radius, material_data_[selected_material]));
				is_changed = true;
			}
		}
		else if (object_type == TRIANGLE)
		{
			static float new_triangle_v0[3]{-1.0f, 0.0f, 0.0f};
			static float new_triangle_v1[3]{0.0f, 1.0f, 0.0f};
			static float new_triangle_v2[3]{1.0f, 0.0f, 0.0f};

			if (ImGui::TreeNode("Properties"))
			{
				ImGui::SliderFloat3("Vertex 0", new_triangle_v0, -UINT16_MAX, UINT16_MAX, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
				ImGui::SliderFloat3("Vertex 1", new_triangle_v1, -UINT16_MAX, UINT16_MAX, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
				ImGui::SliderFloat3("Vertex 2", new_triangle_v2, -UINT16_MAX, UINT16_MAX, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("Material"))
			{
				draw_material_list(material_data_, material_types, selected_material);
				ImGui::TreePop();
			}

			if (ImGui::Button("Create object", {ImGui::GetContentRegionAvail().x, 0}))
			{
				object_data_.push_back(std::make_shared<TriangleInfo>(new_triangle_v0, new_triangle_v1, new_triangle_v2, material_data_[selected_material]));
				is_changed = true;
			}
		}
	}
	return is_changed;
}
