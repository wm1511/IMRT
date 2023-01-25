#include "RtInterface.hpp"
#include "../cpu_renderer/CpuRenderer.hpp"
#include "../cuda_renderer/CudaRenderer.cuh"

#include "../imgui/imgui.h"

#include <chrono>

template <typename T>
static void extend_array(T**& array, const int32_t current_size, int32_t& current_capacity)
{
	if (current_size == current_capacity)
	{
		T** new_array = new T*[(uint64_t)2 * current_size];
		current_capacity *= 2;
		memcpy(new_array, array, current_size * sizeof(T*));
		delete[] array;
		array = new_array;
	}
}

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

static void draw_material_list(MaterialInfo** material_data, const int32_t material_count, const char* material_types[], int32_t& selected_material)
{
	ImGui::BeginChild("Materials", {ImGui::GetContentRegionAvail().x, ImGui::GetFontSize() * 6}, true,
		ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_AlwaysAutoResize);
	ImGui::Text("List of defined materials");
	draw_help("Choose this object's material_id");
	for (int32_t n = 0; n < material_count; n++)
	{
		static char buffer[16]{};
		sprintf_s(buffer, "%u-%s", n, material_types[material_data[n]->type]);
		if (ImGui::Selectable(buffer, selected_material == n))
			selected_material = n;
	}
	ImGui::EndChild();
}

RtInterface::RtInterface()
{
	render_info_.material_data_count = 4;
	render_info_.object_data_count = 4;
	render_info_.material_count = 4;
	render_info_.object_count = 4;
	render_info_.object_capacity = 4;
	render_info_.material_capacity = 4;
	render_info_.material_data = new MaterialInfo*[render_info_.material_count];
	render_info_.object_data = new ObjectInfo*[render_info_.object_count];
	float left[3] = {1.0f, 0.0f, -1.0f};
	float center[3] = {0.0f, 0.0f, -1.0f};
	float right[3] = {-1.0f, 0.0f, -1.0f};
	float ground[3] = {0.0f, -100.5f, -1.0f};
	float gray[3] = {0.5f, 0.5f, 0.5f};
	float blue[3] = {0.2f, 0.2f, 0.8f};
	render_info_.material_data[0] = new DiffuseInfo(gray);
	render_info_.material_data[1] = new RefractiveInfo(1.5f);
	render_info_.material_data[2] = new SpecularInfo(gray, 0.1f);
	render_info_.material_data[3] = new DiffuseInfo(blue);
	render_info_.object_data[0] = new SphereInfo(left, 0.5f, 0);
	render_info_.object_data[1] = new SphereInfo(center, 0.5f, 1);
	render_info_.object_data[2] = new SphereInfo(right, 0.5f, 2);
	render_info_.object_data[3] = new SphereInfo(ground, 100.0f, 3);
}

RtInterface::~RtInterface()
{
	for (int32_t i = 0; i < render_info_.object_data_count; i++)
		delete render_info_.object_data[i];
	delete[] render_info_.object_data;

	for (int32_t i = 0; i < render_info_.material_data_count; i++)
		delete render_info_.material_data[i];
	delete[] render_info_.material_data; 
}

void RtInterface::draw()
{
	//ImGui::ShowDemoWindow();
	{
		ImGui::Begin("Render settings");

		const bool starting_disabled = is_rendering_;
		if (starting_disabled)
			ImGui::BeginDisabled();
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
			renderer_ = std::make_unique<CudaRenderer>(&render_info_);
		}
		ImGui::SameLine();
		if (ImGui::Button("OPTIX render", {ImGui::GetContentRegionAvail().x, 0}))
		{

		}
		if (starting_disabled)
			ImGui::EndDisabled();

		if (ImGui::Button("Stop rendering", {ImGui::GetContentRegionAvail().x, 0}))
		{
			renderer_.reset();
			is_rendering_ = false;
		}

		if (is_rendering_)
		{
			if (!image_ || render_info_.width != image_->GetWidth() || render_info_.height != image_->GetHeight())
			{
				image_ = std::make_unique<Image>(render_info_.width, render_info_.height);
				delete[] image_data_;
				image_data_ = new float[(uint64_t)4 * render_info_.height * render_info_.width];
				renderer_->recreate_image();
				renderer_->recreate_camera();
			}

			const auto start = std::chrono::high_resolution_clock::now();

			renderer_->render(image_data_);

			const auto duration = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start);
			render_time_ = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

			image_->SetData(image_data_);
			frames_rendered_++;
			render_info_.frames_since_refresh++;
		}

		ImGui::Text("Last render time: %llu ms", render_time_);
		ImGui::Text("Frames rendered: %llu", frames_rendered_);
		ImGui::Text("Frames rendered since last refresh: %u", render_info_.frames_since_refresh);

		ImGui::Separator();
		edit_settings();
		edit_camera();
		ImGui::End();
	}

	{
		ImGui::Begin("Scene settings");
		edit_material();
		add_material();
		edit_object();
		add_object();
		ImGui::End();
	}

	{
		ImGui::Begin("Viewport");
		render_info_.width = static_cast<uint32_t>(ImGui::GetContentRegionAvail().x);
		render_info_.height = static_cast<uint32_t>(ImGui::GetContentRegionAvail().y);

		if (image_) 
			ImGui::Image(reinterpret_cast<ImU64>(image_->GetDescriptorSet()),
				{static_cast<float>(image_->GetWidth()), static_cast<float>(image_->GetHeight())},
				ImVec2(1, 0), ImVec2(0, 1));
		ImGui::End();
	}
}

void RtInterface::edit_settings()
{
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
}


void RtInterface::edit_camera()
{
	if (ImGui::CollapsingHeader("Camera settings", ImGuiTreeNodeFlags_DefaultOpen))
    {
		bool is_changed = false;
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
		if (is_rendering_ && is_changed)
		{
			renderer_->recreate_camera();
			render_info_.frames_since_refresh = 0;
		}
	}
}

void RtInterface::add_material()
{
	static const char* material_types[]{"Unknown_Material", "Diffuse", "Specular", "Refractive"};
	
	if (ImGui::CollapsingHeader("Add material"))
	{
		bool is_changed = false;
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
				extend_array(render_info_.material_data, render_info_.material_data_count, render_info_.material_capacity);
				render_info_.material_data[render_info_.material_data_count] = new DiffuseInfo(new_diffuse_color);
				render_info_.material_data_count++;
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
				extend_array(render_info_.material_data, render_info_.material_data_count, render_info_.material_capacity);
				render_info_.material_data[render_info_.material_data_count] = new SpecularInfo(new_specular_color, new_specular_fuzziness);
				render_info_.material_data_count++;
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
				extend_array(render_info_.material_data, render_info_.material_data_count, render_info_.material_capacity);
				render_info_.material_data[render_info_.material_data_count] = new RefractiveInfo(new_refractive_index_of_refraction);
				render_info_.material_data_count++;
				is_changed = true;
			}
		}
		if (is_rendering_ && is_changed)
		{
			renderer_->recreate_world();
			render_info_.material_count++;
			render_info_.frames_since_refresh = 0;
		}
	}
}

void RtInterface::edit_material()
{
	static const char* material_types[]{"Unknown_Material", "Diffuse", "Specular", "Refractive"};

	if (ImGui::CollapsingHeader("Material list"))
	{
		bool is_changed = false;
		for (int32_t i = 0; i < render_info_.material_count; i++)
		{
			MaterialInfo* current_material = render_info_.material_data[i];
			ImGui::PushID(i);
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
						const auto current_diffuse = (DiffuseInfo*)current_material;
						is_changed |= ImGui::ColorEdit3("Color", current_diffuse->albedo, ImGuiColorEditFlags_Float);
					}
					else if (current_material->type == SPECULAR)
					{
						const auto current_specular = (SpecularInfo*)current_material;
						is_changed |= ImGui::ColorEdit3("Color", current_specular->albedo, ImGuiColorEditFlags_Float);
						is_changed |= ImGui::SliderFloat("Fuzziness", &current_specular->fuzziness, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
						draw_help("Fuzziness of material reflection");
					}
					else if (current_material->type == REFRACTIVE)
					{
						const auto current_refractive = (RefractiveInfo*)current_material;
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
		if (is_rendering_ && is_changed)
		{
			renderer_->recreate_world();
			render_info_.frames_since_refresh = 0;
		}
	}
}

void RtInterface::add_object()
{
	static const char* object_types[]{"Unknown_Object", "Sphere", "Triangle", "Triangle_Mesh"};
	static const char* material_types[]{"Unknown_Material", "Diffuse", "Specular", "Refractive"};

	if (ImGui::CollapsingHeader("Add object"))
	{
		bool is_changed = false;
		static int32_t object_type = UNKNOWN_OBJECT;
		static int32_t selected_material = 0;

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
				ImGui::SliderFloat("Radius", &new_sphere_radius, -UINT8_MAX, UINT8_MAX, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
				draw_help("Radius of sphere can be negative for refractive spheres to make sphere hollow");
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("Material"))
			{
				draw_material_list(render_info_.material_data, render_info_.material_count, material_types, selected_material);
				ImGui::TreePop();
			}

			if (ImGui::Button("Create object", {ImGui::GetContentRegionAvail().x, 0}))
			{
				extend_array(render_info_.object_data, render_info_.object_data_count, render_info_.object_capacity);
				render_info_.object_data[render_info_.object_data_count] = new SphereInfo(new_sphere_center, new_sphere_radius, selected_material);
				render_info_.object_data_count++;
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
				draw_material_list(render_info_.material_data, render_info_.material_count, material_types, selected_material);
				ImGui::TreePop();
			}

			if (ImGui::Button("Create object", {ImGui::GetContentRegionAvail().x, 0}))
			{
				extend_array(render_info_.object_data, render_info_.object_data_count, render_info_.object_capacity);
				render_info_.object_data[render_info_.object_data_count] = new TriangleInfo(new_triangle_v0, new_triangle_v1, new_triangle_v2, selected_material);
				render_info_.object_data_count++;
				is_changed = true;
			}
		}
		else if (object_type == TRIANGLE_MESH)
		{
			ImGui::Text("Not implemented");
		}
		if (is_rendering_ && is_changed)
		{
			renderer_->recreate_world();
			render_info_.object_count++;
			render_info_.frames_since_refresh = 0;
		}
	}
}

void RtInterface::edit_object()
{
	static const char* material_types[]{"Unknown_Material", "Diffuse", "Specular", "Refractive"};
	static const char* object_types[]{"Unknown_Object", "Sphere", "Triangle", "Triangle_Mesh"};

	if (ImGui::CollapsingHeader("Object list"))
	{
		bool is_changed = false;
		for (int32_t i = 0; i < render_info_.object_count; i++)
		{
			ObjectInfo* current_object = render_info_.object_data[i];

			ImGui::PushID(i);
			if (ImGui::TreeNode("Object", "%u-%s-%s", i, object_types[current_object->type], material_types[render_info_.material_data[current_object->material_id]->type]))
			{
				if (ImGui::TreeNode("Properties"))
				{
					ImGui::Text("Object type: %s", object_types[current_object->type]);
					if (current_object->type == UNKNOWN_OBJECT)
					{
						ImGui::Text("Object properties can't be set for type Unknown_Object");
					}
					else if (current_object->type == SPHERE)
					{
						const auto current_sphere = (SphereInfo*)current_object;

						is_changed |= ImGui::SliderFloat3("Center", current_sphere->center, -UINT8_MAX, UINT8_MAX,"%.3f",
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
						is_changed |= ImGui::SliderFloat("Radius", &current_sphere->radius, -UINT16_MAX, UINT16_MAX, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
						draw_help("Radius of sphere can be negative for refractive spheres to make sphere hollow");
					}
					else if (current_object->type == TRIANGLE)
					{
						const auto current_triangle = (TriangleInfo*)current_object;

						is_changed |= ImGui::SliderFloat3("Vertex 0", current_triangle->v0, -UINT16_MAX, UINT16_MAX, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
						is_changed |= ImGui::SliderFloat3("Vertex 1", current_triangle->v1, -UINT16_MAX, UINT16_MAX, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
						is_changed |= ImGui::SliderFloat3("Vertex 2", current_triangle->v2, -UINT16_MAX, UINT16_MAX, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
					}
					ImGui::TreePop();
				}

				if (current_object->type == TRIANGLE_MESH)
				{
					if (ImGui::TreeNode("Transform"))
					{
						const auto current_triangle_mesh = (TriangleMeshInfo*)current_object;

						is_changed |= ImGui::SliderFloat3("Translation", current_triangle_mesh->transform.translation, -UINT16_MAX, UINT16_MAX, "%.3f", 
								ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_Logarithmic);
						is_changed |= ImGui::SliderFloat3("Scale", current_triangle_mesh->transform.scale, 0.0f, UINT8_MAX, "%.3f", 
								ImGuiSliderFlags_AlwaysClamp);
						is_changed |= ImGui::SliderAngle("Rotation x", &current_triangle_mesh->transform.rotation[0], 0.0f, 360.0f, "%.3f", 
								ImGuiSliderFlags_AlwaysClamp);
						is_changed |= ImGui::SliderAngle("Rotation y", &current_triangle_mesh->transform.rotation[1], 0.0f, 360.0f, "%.3f", 
								ImGuiSliderFlags_AlwaysClamp);
						is_changed |= ImGui::SliderAngle("Rotation z", &current_triangle_mesh->transform.rotation[2], 0.0f, 360.0f, "%.3f", 
								ImGuiSliderFlags_AlwaysClamp);
						ImGui::TreePop();
					}
				}

				if (ImGui::TreeNode("Material"))
				{
					ImGui::Text("Object's material id: %u", current_object->material_id);
					static int32_t selected_material = 0;
		            draw_material_list(render_info_.material_data, render_info_.material_count, material_types, selected_material);

					if (ImGui::Button("Set material", {ImGui::GetContentRegionAvail().x, 0}))
					{
						current_object->material_id = selected_material;
						is_changed = true;
					}
					ImGui::TreePop();
				}

				if (ImGui::Button("Delete object", {ImGui::GetContentRegionAvail().x, 0}))
				{
					delete render_info_.object_data[i];
					if (render_info_.object_count > 1 && render_info_.object_data[render_info_.object_count - 1] != render_info_.object_data[i])
						render_info_.object_data[i] = render_info_.object_data[render_info_.object_count - 1];
					render_info_.object_count--;
					render_info_.object_data_count--;
					is_changed = true;
				}

				ImGui::TreePop();
			}
			ImGui::PopID();
		}
		if (is_rendering_ && is_changed)
		{
			renderer_->recreate_world();
			render_info_.frames_since_refresh = 0;
		}
	}
}
