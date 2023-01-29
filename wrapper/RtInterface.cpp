#include "RtInterface.hpp"
#include "../cpu_renderer/CpuRenderer.hpp"
#include "../cuda_renderer/CudaRenderer.cuh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "../imgui/imgui.h"

#include <filesystem>
#include <chrono>

template <typename T>
static void extend_array(T**& array, const int32_t current_size, int32_t& current_capacity)
{
	if (current_size == current_capacity)
	{
		T** new_array = new T*[(uint64_t)2 * current_size];
		current_capacity *= 2;
		memcpy_s(new_array, current_capacity * sizeof(T*), array, current_size * sizeof(T*));
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
	render_info_.material_data[0] = new DiffuseInfo({0.5f, 0.5f, 0.5f});
	render_info_.material_data[1] = new RefractiveInfo(1.5f);
	render_info_.material_data[2] = new SpecularInfo({0.5f, 0.5f, 0.5f}, 0.1f);
	render_info_.material_data[3] = new DiffuseInfo({0.2f, 0.2f, 0.8f});
	render_info_.object_data[0] = new SphereInfo({1.0f, 0.0f, -1.0f}, 0.5f, 0);
	render_info_.object_data[1] = new SphereInfo({0.0f, 0.0f, -1.0f}, 0.5f, 1);
	render_info_.object_data[2] = new SphereInfo({-1.0f, 0.0f, -1.0f}, 0.5f, 2);
	render_info_.object_data[3] = new SphereInfo({0.0f, -100.5f, -1.0f}, 100.0f, 3);
}

RtInterface::~RtInterface()
{
	for (int32_t i = 0; i < render_info_.object_data_count; i++)
		delete render_info_.object_data[i];
	delete[] render_info_.object_data;

	for (int32_t i = 0; i < render_info_.material_data_count; i++)
		delete render_info_.material_data[i];
	delete[] render_info_.material_data;

	stbi_image_free(render_info_.hdr_data);
}

void RtInterface::draw()
{
	ImGui::ShowDemoWindow();

	const auto start = std::chrono::high_resolution_clock::now();
	{
		ImGui::Begin("Render settings");

		const bool starting_disabled = is_rendering_;
		if (starting_disabled)
			ImGui::BeginDisabled();
		if (ImGui::Button("CPU render", {ImGui::GetContentRegionAvail().x / 3, 0}))
		{
			frames_rendered_ = 0;
			render_info_.frames_since_refresh = 0;
			is_rendering_ = true;
			renderer_ = std::make_unique<CpuRenderer>(&render_info_);
		}
		ImGui::SameLine();
		if (ImGui::Button("CUDA render", {ImGui::GetContentRegionAvail().x / 2, 0}))
		{
			frames_rendered_ = 0;
			render_info_.frames_since_refresh = 0;
			is_rendering_ = true;
			renderer_ = std::make_unique<CudaRenderer>(&render_info_);
		}
		ImGui::SameLine();
		if (ImGui::Button("OPTIX render", {ImGui::GetContentRegionAvail().x, 0}))
		{

		}
		if (starting_disabled)
			ImGui::EndDisabled();

		if (ImGui::Button("Stop rendering", {ImGui::GetContentRegionAvail().x, 0}) || (render_info_.render_mode == STATIC && frames_rendered_ != 0))
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
				renderer_->refresh_buffer();
				render_info_.frames_since_refresh = 0;
			}

			frames_rendered_++;
			render_info_.frames_since_refresh++;

			renderer_->render(image_data_);

			image_->SetData(image_data_);
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
		edit_sky();
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

		if (ImGui::IsWindowFocused())
			move_camera();

		ImGui::End();
	}

	const auto duration = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start);
	render_time_ = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

void RtInterface::move_camera()
{
	bool is_changed = false;

	if (ImGui::IsKeyDown(ImGuiKey_W))
	{
		render_info_.camera_position -= render_info_.camera_direction * camera_movement_speed_;
		is_changed = true;
	}
	if (ImGui::IsKeyDown(ImGuiKey_S))
	{
		render_info_.camera_position += render_info_.camera_direction * camera_movement_speed_;
		is_changed = true;
	}
	if (ImGui::IsKeyDown(ImGuiKey_A))
	{
		const float3 displacement = normalize(cross(render_info_.camera_direction, make_float3(0.0f, -1.0f, 0.0f)));
		render_info_.camera_position -= displacement * camera_movement_speed_;
		is_changed = true;
	}
	if (ImGui::IsKeyDown(ImGuiKey_D))
	{
		const float3 displacement = normalize(cross(render_info_.camera_direction, make_float3(0.0f, -1.0f, 0.0f)));
		render_info_.camera_position += displacement * camera_movement_speed_;
		is_changed = true;
	}
	if (ImGui::IsKeyDown(ImGuiKey_E))
	{
		render_info_.camera_position.y += camera_movement_speed_;
		is_changed = true;
	}
	if (ImGui::IsKeyDown(ImGuiKey_Q))
	{
		render_info_.camera_position.y -= camera_movement_speed_;
		is_changed = true;
	}

	if (ImGui::IsMouseDragging(ImGuiMouseButton_Left))
	{
		render_info_.angle_x += ImGui::GetMouseDragDelta().x * camera_rotation_speed_;
		render_info_.angle_y += ImGui::GetMouseDragDelta().y * camera_rotation_speed_;
		render_info_.angle_x = fmodf(render_info_.angle_x, kTwoPi);
		render_info_.angle_y = clamp(render_info_.angle_y, -kHalfPi, kHalfPi);
		render_info_.camera_direction = normalize(make_float3(
			cos(render_info_.angle_y) * -sin(render_info_.angle_x),
			-sin(render_info_.angle_y),
			-cos(render_info_.angle_x) * cos(render_info_.angle_y)));
		ImGui::ResetMouseDragDelta();
		is_changed =  true;
	}
	
	if (is_rendering_ && is_changed)
	{
		renderer_->refresh_camera();
		renderer_->refresh_buffer();
		render_info_.frames_since_refresh = 0;
	}
}

void RtInterface::edit_settings()
{
	if (ImGui::CollapsingHeader("Quality settings", ImGuiTreeNodeFlags_DefaultOpen))
    {
		bool is_changed = false;
		is_changed |= ImGui::RadioButton("Progressive render", &render_info_.render_mode, PROGRESSIVE); ImGui::SameLine();
        is_changed |= ImGui::RadioButton("Static render", &render_info_.render_mode, STATIC);

		if (render_info_.render_mode == PROGRESSIVE)
			ImGui::BeginDisabled();
		if (ImGui::TreeNode("Samples per pixel"))
		{
			ImGui::SliderInt("##SamplesPerPixel", &render_info_.samples_per_pixel, 1, INT16_MAX, "%d",
				ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
			draw_help("Count of samples generated for each pixel");
			ImGui::TreePop();
		}
		if (render_info_.render_mode == PROGRESSIVE)
			ImGui::EndDisabled();

		if (ImGui::TreeNode("Recursion depth"))
		{
			is_changed |= ImGui::SliderInt("##RecursionDepth", &render_info_.max_depth, 1, INT8_MAX, "%d", ImGuiSliderFlags_AlwaysClamp);
			draw_help("Maximum depth, that recursion can achieve before being stopped");
			ImGui::TreePop();
		}
		if (is_rendering_ && is_changed)
		{
			renderer_->refresh_buffer();
			render_info_.frames_since_refresh = 0;
		}
    }
}

void RtInterface::edit_camera()
{
	if (ImGui::CollapsingHeader("Camera settings", ImGuiTreeNodeFlags_DefaultOpen))
    {
		bool is_changed = false;
	    if (ImGui::TreeNode("Position"))
		{
			is_changed |= ImGui::SliderFloat("##CameraX", &render_info_.camera_position.x, -UINT16_MAX, UINT16_MAX, "%.3f",
				ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
			ImGui::SameLine(); ImGui::Text("x");
	    	is_changed |= ImGui::SliderFloat("##CameraY", &render_info_.camera_position.y, -UINT16_MAX, UINT16_MAX, "%.3f",
				ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
			ImGui::SameLine(); ImGui::Text("y");
	    	is_changed |= ImGui::SliderFloat("##CameraZ", &render_info_.camera_position.z, -UINT16_MAX, UINT16_MAX, "%.3f",
				ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
			ImGui::SameLine(); ImGui::Text("z");
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("Angle offset"))
		{
			is_changed |= ImGui::SliderAngle("degrees x", &render_info_.angle_x, 0.0f, 360.0f, "%.3f");
			is_changed |= ImGui::SliderAngle("degrees y", &render_info_.angle_y, -90.0f, 90.0f, "%.3f");

			render_info_.camera_direction = normalize(make_float3(
			cos(render_info_.angle_y) * -sin(render_info_.angle_x),
			-sin(render_info_.angle_y),
			-cos(render_info_.angle_x) * cos(render_info_.angle_y)));
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("Vertical field of view"))
		{
			is_changed |= ImGui::SliderAngle("degrees", &render_info_.fov, 0.0f, 180.0f, "%.3f");
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("Aperture"))
		{
			is_changed |= ImGui::SliderFloat("##Aperture", &render_info_.aperture, 0.0f, UINT8_MAX, "%.3f",
				ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("Focus distance"))
		{
			is_changed |= ImGui::SliderFloat("##FocusDist", &render_info_.focus_distance, 0.0f, UINT8_MAX, "%.3f",
				ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("Movement speed"))
		{
			ImGui::SliderFloat("##MoveSpeed", &camera_movement_speed_, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("Rotation speed"))
		{
			ImGui::SliderFloat("##RotateSpeed", &camera_rotation_speed_, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
			ImGui::TreePop();
		}

		if (is_rendering_ && is_changed)
		{
			renderer_->refresh_camera();
			renderer_->refresh_buffer();
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
				render_info_.material_data[render_info_.material_data_count] = new DiffuseInfo(make_float3(new_diffuse_color));
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
				render_info_.material_data[render_info_.material_data_count] = new SpecularInfo(make_float3(new_specular_color), new_specular_fuzziness);
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
			renderer_->refresh_buffer();
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
						is_changed |= ImGui::ColorEdit3("Color", current_diffuse->albedo_array, ImGuiColorEditFlags_Float);
					}
					else if (current_material->type == SPECULAR)
					{
						const auto current_specular = (SpecularInfo*)current_material;
						is_changed |= ImGui::ColorEdit3("Color", current_specular->albedo_array, ImGuiColorEditFlags_Float);
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
			renderer_->refresh_world();
			renderer_->refresh_buffer();
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
				render_info_.object_data[render_info_.object_data_count] = new SphereInfo(
					make_float3(new_sphere_center), new_sphere_radius, selected_material);
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
				render_info_.object_data[render_info_.object_data_count] = new TriangleInfo(
					make_float3(new_triangle_v0), make_float3(new_triangle_v1), make_float3(new_triangle_v2), selected_material);
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
			renderer_->refresh_buffer();
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

						is_changed |= ImGui::SliderFloat3("Center", current_sphere->center_array, -UINT8_MAX, UINT8_MAX,"%.3f",
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
						is_changed |= ImGui::SliderFloat("Radius", &current_sphere->radius, -UINT16_MAX, UINT16_MAX, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
						draw_help("Radius of sphere can be negative for refractive spheres to make sphere hollow");
					}
					else if (current_object->type == TRIANGLE)
					{
						const auto current_triangle = (TriangleInfo*)current_object;

						is_changed |= ImGui::SliderFloat3("Vertex 0", current_triangle->v0_array, -UINT16_MAX, UINT16_MAX, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
						is_changed |= ImGui::SliderFloat3("Vertex 1", current_triangle->v1_array, -UINT16_MAX, UINT16_MAX, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
						is_changed |= ImGui::SliderFloat3("Vertex 2", current_triangle->v2_array, -UINT16_MAX, UINT16_MAX, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
					}
					ImGui::TreePop();
				}

				/*if (current_object->type == TRIANGLE_MESH)
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
				}*/

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
			renderer_->refresh_world();
			renderer_->refresh_buffer();
			render_info_.frames_since_refresh = 0;
		}
	}
}

void RtInterface::edit_sky()
{
	if (ImGui::CollapsingHeader("Environment map"))
	{
		bool hdr_changed = false, exposure_changed = false;
		static char selected_file_path[256];
		static char file_label[128];
		static wchar_t selected_file[128];

		const auto iterator = std::filesystem::recursive_directory_iterator("hdr/");
		uint64_t converted_count{};

		ImGui::BeginChild("HDR Files", {ImGui::GetContentRegionAvail().x, ImGui::GetFontSize() * 6}, true,
			ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_AlwaysAutoResize);
		ImGui::Text("Choose HDR image");
		draw_help("Choose file containing HDR image for environment map creation. New files can be added to \"hdr\" folder");

		for (const auto& entry : iterator)
		{
			wcstombs_s(&converted_count, file_label, entry.path().filename().c_str(), 128);
			if (ImGui::Selectable(file_label, wcscmp(entry.path().filename().c_str(), selected_file) == 0))
			{
				wcscpy_s(selected_file, 128, entry.path().filename().c_str());
				const auto path = std::filesystem::current_path() / L"hdr" / entry.path().filename().c_str();
				wcstombs_s(&converted_count, selected_file_path, path.c_str(), 256);
			}
		}
		ImGui::EndChild();

		exposure_changed |= ImGui::SliderFloat("Exposure", &render_info_.hdr_exposure, 0.0f, 16.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
		draw_help("Adjust brightness of HDR environment map");

		ImGui::TextColored({1.0f, 0.0f, 0.0f, 1.0f}, "Loading large files will take a while");

		if (ImGui::Button("Set HDR", {ImGui::GetContentRegionAvail().x, 0}))
		{
			if (!stbi_info(selected_file_path, &render_info_.hdr_width, &render_info_.hdr_height, &render_info_.hdr_components))
			{
				ImGui::OpenPopup("HDR loading failed");
			}
			else
			{
				stbi_image_free(render_info_.hdr_data);
				render_info_.hdr_data = stbi_loadf(selected_file_path, &render_info_.hdr_width, &render_info_.hdr_height, &render_info_.hdr_components, 0);
				hdr_changed = true;
			}
		}

		ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, {0.5f, 0.5f});
		if (ImGui::BeginPopupModal("HDR loading failed", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
		{
			ImGui::Text("%s is not a valid HDR image file", selected_file_path);
			if (ImGui::Button("OK", ImVec2(ImGui::GetContentRegionAvail().x, 0))) 
				ImGui::CloseCurrentPopup();

			ImGui::EndPopup();
		}

		if (ImGui::Button("Clear HDR", {ImGui::GetContentRegionAvail().x, 0}))
		{
			stbi_image_free(render_info_.hdr_data);
			render_info_.hdr_data = nullptr;
			hdr_changed = true;
		}
		if (is_rendering_)
		{
			if (exposure_changed)
			{
				renderer_->refresh_buffer();
				render_info_.frames_since_refresh = 0;
			}
			else if (hdr_changed)
			{
				renderer_->recreate_sky();
				renderer_->refresh_buffer();
				render_info_.frames_since_refresh = 0;
			}
		}
	}
}
