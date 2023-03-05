#include "stdafx.h"
#include "RtInterface.hpp"
#include "../cpu_renderer/CpuRenderer.hpp"
#include "../cuda_renderer/CudaRenderer.cuh"

#include "stb_image.h"
#include "stb_image_write.h"
#include "../imgui/imgui.h"

// Intellisense doesn't work without this include
#include <filesystem>

static const char* object_types[]{"Unknown Object", "Sphere", "Triangle", "Plane", "Volumetric Sphere", "Cylinder", "Cone", "Torus", "Model"};
static const char* material_types[]{"Unknown Material", "Diffuse", "Specular", "Refractive", "Isotropic"};
static const char* texture_types[]{"Unknown Texture", "Solid", "Image", "Checker"};

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

static void draw_textures(TextureInfo** texture_data, const int32_t texture_count, int32_t& selected_texture)
{
	ImGui::BeginChild("Textures", {ImGui::GetContentRegionAvail().x, ImGui::GetFontSize() * 6}, true,
		ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_AlwaysAutoResize);
	ImGui::Text("List of defined textures");
	draw_help("Choose this object's texture id");
	for (int32_t n = 0; n < texture_count; n++)
	{
		static char buffer[64]{};
		if (sprintf_s(buffer, "%i-%s", n, texture_types[texture_data[n]->type]) > 0)
		{
			if (ImGui::Selectable(buffer, selected_texture == n))
				selected_texture = n;
		}
		else
			ImGui::Text("Failed to print texture identifier");
	}
	ImGui::EndChild();
}

static void draw_materials(TextureInfo** texture_data, MaterialInfo** material_data, const int32_t material_count, int32_t& selected_material)
{
	ImGui::BeginChild("Materials", {ImGui::GetContentRegionAvail().x, ImGui::GetFontSize() * 6}, true,
		ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_AlwaysAutoResize);
	ImGui::Text("List of defined materials");
	draw_help("Choose this object's material id");
	for (int32_t n = 0; n < material_count; n++)
	{
		static char buffer[64]{};
		if (sprintf_s(buffer, "%i-%s-%s", n, material_types[material_data[n]->type], texture_types[texture_data[material_data[n]->texture_id]->type]) > 0)
		{
			if (ImGui::Selectable(buffer, selected_material == n))
				selected_material = n;
		}
		else
			ImGui::Text("Failed to print material identifier");
	}
	ImGui::EndChild();
}

static void draw_files(std::filesystem::path& selected_file, const char* id, const char* directory)
{
	ImGui::BeginChild(id, {ImGui::GetContentRegionAvail().x, ImGui::GetFontSize() * 6}, true,
			ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_AlwaysAutoResize);
	const auto iterator = std::filesystem::recursive_directory_iterator(directory);

	for (const auto& entry : iterator)
		if (ImGui::Selectable(entry.path().filename().u8string().c_str(), entry.path() == selected_file))
			selected_file = entry.path();
			
	ImGui::EndChild();

	ImGui::TextColored({1.0f, 0.0f, 0.0f, 1.0f}, "Loading large files will take a while");
}

static void draw_modal(const char* id, const char* message)
{
	ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, {0.5f, 0.5f});
	if (ImGui::BeginPopupModal(id, nullptr, ImGuiWindowFlags_AlwaysAutoResize))
	{
		ImGui::Text(message);
		if (ImGui::Button("OK", ImVec2(ImGui::GetContentRegionAvail().x, 0)))
			ImGui::CloseCurrentPopup();

		ImGui::EndPopup();
	}
}

RtInterface::RtInterface()
{
	sky_info_.create_sky();
}

RtInterface::~RtInterface()
{
	sky_info_.clear_hdr();
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
		if (ImGui::Button("CPU render", {ImGui::GetContentRegionAvail().x / 2, 0}))
		{
			frames_rendered_ = 0;
			render_info_.frames_since_refresh = 0;
			is_rendering_ = true;
			renderer_ = std::make_unique<CpuRenderer>(&render_info_, &world_info_, &sky_info_);
		}
		ImGui::SameLine();
		if (ImGui::Button("CUDA render", {ImGui::GetContentRegionAvail().x, 0}))
		{
			frames_rendered_ = 0;
			render_info_.frames_since_refresh = 0;
			is_rendering_ = true;
			renderer_ = std::make_unique<CudaRenderer>(&render_info_, &world_info_, &sky_info_);
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
			if (!image_ || render_info_.width != image_->get_width() || render_info_.height != image_->get_height())
			{
				image_ = std::make_unique<Frame>(render_info_.width, render_info_.height);
				delete[] image_data_;
				image_data_ = new float[(uint64_t)4 * render_info_.height * render_info_.width];
				renderer_->recreate_image();
				renderer_->recreate_camera();
				renderer_->refresh_buffer();
				render_info_.frames_since_refresh = 0;
			}

			frames_rendered_++;
			render_info_.frames_since_refresh++;

			render_info_.frame_needs_display = fmod(log2f((float)render_info_.frames_since_refresh), 1) == 0.0f;

			renderer_->render(image_data_);

			if (render_info_.frame_needs_display)
				image_->set_data(image_data_);
		}

		ImGui::Text("Last render time: %llu ms", render_time_);
		ImGui::Text("Frames rendered: %llu", frames_rendered_);
		ImGui::Text("Frames rendered since last refresh: %u", render_info_.frames_since_refresh);

		ImGui::Separator();
		edit_settings();
		edit_camera();
		save_image();
		ImGui::End();
	}

	{
		ImGui::Begin("Scene settings");
		edit_texture();
		add_texture();
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
			ImGui::Image(reinterpret_cast<ImU64>(image_->get_descriptor_set()),
				{static_cast<float>(image_->get_width()), static_cast<float>(image_->get_height())},
				ImVec2(1, 0), ImVec2(0, 1));

		if (is_rendering_ && ImGui::IsWindowFocused())
			move_camera();

		ImGui::End();
	}

	const auto duration = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start);
	if (is_rendering_)
		render_time_ = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

void RtInterface::move_camera()
{
	bool is_moved = false;

	if (ImGui::IsKeyDown(ImGuiKey_W))
	{
		render_info_.camera_position -= render_info_.camera_direction * camera_movement_speed_ * (float)render_time_;
		is_moved = true;
	}
	if (ImGui::IsKeyDown(ImGuiKey_S))
	{
		render_info_.camera_position += render_info_.camera_direction * camera_movement_speed_ * (float)render_time_;
		is_moved = true;
	}
	if (ImGui::IsKeyDown(ImGuiKey_A))
	{
		const float3 displacement = normalize(cross(render_info_.camera_direction, make_float3(0.0f, -1.0f, 0.0f)));
		render_info_.camera_position -= displacement * camera_movement_speed_ * (float)render_time_;
		is_moved = true;
	}
	if (ImGui::IsKeyDown(ImGuiKey_D))
	{
		const float3 displacement = normalize(cross(render_info_.camera_direction, make_float3(0.0f, -1.0f, 0.0f)));
		render_info_.camera_position += displacement * camera_movement_speed_ * (float)render_time_;
		is_moved = true;
	}
	if (ImGui::IsKeyDown(ImGuiKey_E))
	{
		render_info_.camera_position.y += camera_movement_speed_ * (float)render_time_;
		is_moved = true;
	}
	if (ImGui::IsKeyDown(ImGuiKey_Q))
	{
		render_info_.camera_position.y -= camera_movement_speed_ * (float)render_time_;
		is_moved = true;
	}

	if (ImGui::IsMouseDragging(ImGuiMouseButton_Left))
	{
		render_info_.angle_x += ImGui::GetMouseDragDelta().x * camera_rotation_speed_;
		render_info_.angle_y += ImGui::GetMouseDragDelta().y * camera_rotation_speed_;
		render_info_.angle_x = fmodf(render_info_.angle_x, kTwoPi);
		render_info_.angle_y = clamp(render_info_.angle_y, -kHalfPi, kHalfPi);
		render_info_.camera_direction = make_float3(
			cos(render_info_.angle_y) * -sin(render_info_.angle_x),
			-sin(render_info_.angle_y),
			-cos(render_info_.angle_x) * cos(render_info_.angle_y));
		ImGui::ResetMouseDragDelta();
		is_moved =  true;
	}
	
	if (is_moved)
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
		bool is_edited = false;
		is_edited |= ImGui::RadioButton("Progressive render", &render_info_.render_mode, PROGRESSIVE); ImGui::SameLine();
        is_edited |= ImGui::RadioButton("Static render", &render_info_.render_mode, STATIC);

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
			is_edited |= ImGui::SliderInt("##RecursionDepth", &render_info_.max_depth, 1, INT8_MAX, "%d", ImGuiSliderFlags_AlwaysClamp);
			draw_help("Maximum depth, that recursion can achieve before being stopped");
			ImGui::TreePop();
		}

		if (is_rendering_ && is_edited)
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
		bool is_edited = false;
	    if (ImGui::TreeNode("Position"))
		{
			is_edited |= ImGui::SliderFloat("##CameraX", &render_info_.camera_position.x, -UINT16_MAX, UINT16_MAX, "%.3f",
				ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
			ImGui::SameLine(); ImGui::Text("x");
	    	is_edited |= ImGui::SliderFloat("##CameraY", &render_info_.camera_position.y, -UINT16_MAX, UINT16_MAX, "%.3f",
				ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
			ImGui::SameLine(); ImGui::Text("y");
	    	is_edited |= ImGui::SliderFloat("##CameraZ", &render_info_.camera_position.z, -UINT16_MAX, UINT16_MAX, "%.3f",
				ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
			ImGui::SameLine(); ImGui::Text("z");
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("Angle offset"))
		{
			is_edited |= ImGui::SliderAngle("degrees x", &render_info_.angle_x, 0.0f, 360.0f, "%.3f");
			is_edited |= ImGui::SliderAngle("degrees y", &render_info_.angle_y, -90.0f, 90.0f, "%.3f");

			render_info_.camera_direction = normalize(make_float3(
			cos(render_info_.angle_y) * -sin(render_info_.angle_x),
			-sin(render_info_.angle_y),
			-cos(render_info_.angle_x) * cos(render_info_.angle_y)));
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("Vertical field of view"))
		{
			is_edited |= ImGui::SliderAngle("degrees", &render_info_.fov, 0.0f, 180.0f, "%.3f");
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("Aperture"))
		{
			is_edited |= ImGui::SliderFloat("##Aperture", &render_info_.aperture, 0.0f, UINT8_MAX, "%.3f",
				ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("Focus distance"))
		{
			is_edited |= ImGui::SliderFloat("##FocusDist", &render_info_.focus_distance, 0.0f, UINT8_MAX, "%.3f",
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

		if (is_rendering_ && is_edited)
		{
			renderer_->refresh_camera();
			renderer_->refresh_buffer();
			render_info_.frames_since_refresh = 0;
		}
	}
}

void RtInterface::add_texture()
{
	if (ImGui::CollapsingHeader("Add texture"))
	{
		bool is_added = false;
		static int32_t texture_type = UNKNOWN_TEXTURE;

		ImGui::Combo("Texture type", &texture_type, texture_types, IM_ARRAYSIZE(texture_types));
		if (texture_type == UNKNOWN_MATERIAL)
		{
			ImGui::Text("Texture of type Unknown_Texture can't be instantiated");
		}
		else if (texture_type == SOLID)
		{
			static Float3 new_solid_color{0.0f, 0.0f, 0.0f};

			if (ImGui::TreeNode("Properties"))
			{
				ImGui::ColorEdit3("Color", new_solid_color.arr, ImGuiColorEditFlags_Float);
				ImGui::TreePop();
			}

			if (ImGui::Button("Create texture", {ImGui::GetContentRegionAvail().x, 0}))
			{
				if (is_rendering_)
					renderer_->deallocate_world();
				world_info_.add_texture(new SolidInfo(new_solid_color.str));
				is_added = true;
			}
		}
		else if (texture_type == IMAGE)
		{
			static int32_t new_image_width{0}, new_image_height{0}, new_image_components{0};
			static std::filesystem::path selected_file;

			ImGui::Text("Choose image");
			draw_help("Choose image file for texture creation. New files can be added to \"assets/tex\" folder");

			draw_files(selected_file, "Texture files", "assets/tex/");

			if (ImGui::Button("Create texture", {ImGui::GetContentRegionAvail().x, 0}))
			{
				if (!stbi_info(selected_file.u8string().c_str(), &new_image_width, &new_image_height, &new_image_components))
				{
					ImGui::OpenPopup("Texture loading failed");
				}
				else
				{
					float* new_image_data = stbi_loadf(selected_file.u8string().c_str(), &new_image_width, &new_image_height, &new_image_components, 3);
					if (is_rendering_)
						renderer_->deallocate_world();
					world_info_.add_texture(new ImageInfo(new_image_data, new_image_width, new_image_height));
					is_added = true;
				}
			}

			draw_modal("Texture loading failed", "This file can't be loaded as image");
		}
		else if (texture_type == CHECKER)
		{
			static Float3 new_checker_color_a{0.0f, 0.0f, 0.0f};
			static Float3 new_checker_color_b{0.0f, 0.0f, 0.0f};
			static float new_checker_tile_size{0.1f};

			if (ImGui::TreeNode("Properties"))
			{
				ImGui::ColorEdit3("Color 1", new_checker_color_a.arr, ImGuiColorEditFlags_Float);
				ImGui::ColorEdit3("Color 2", new_checker_color_b.arr, ImGuiColorEditFlags_Float);
				ImGui::SliderFloat("Tile size", &new_checker_tile_size, 0.001f, UINT8_MAX, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
				ImGui::TreePop();
			}

			if (ImGui::Button("Create texture", {ImGui::GetContentRegionAvail().x, 0}))
			{
				if (is_rendering_)
					renderer_->deallocate_world();
				world_info_.add_texture(new CheckerInfo(new_checker_color_a.str, new_checker_color_b.str, new_checker_tile_size));
				is_added = true;
			}
		}

		if (is_rendering_ && is_added)
		{
			renderer_->allocate_world();
			renderer_->refresh_buffer();
			render_info_.frames_since_refresh = 0;
		}
	}
}

void RtInterface::edit_texture()
{
	if (ImGui::CollapsingHeader("Texture list"))
	{
		bool is_edited = false;
		for (int32_t i = 0; i < world_info_.textures_.size(); i++)
		{
			TextureInfo* current_texture = world_info_.textures_[i];
			ImGui::PushID(i);
			if (ImGui::TreeNode("Texture", "%u-%s", i, texture_types[current_texture->type]))
			{
				ImGui::Text("Texture type: %s", texture_types[current_texture->type]);

				if (ImGui::TreeNode("Properties"))
				{
					if (current_texture->type == UNKNOWN_TEXTURE)
					{
						ImGui::Text("Texture properties can't be set for type Unknown_Texture");
					}
					else if (current_texture->type == SOLID)
					{
						const auto current_solid = (SolidInfo*)current_texture;
						is_edited |= ImGui::ColorEdit3("Color", current_solid->albedo.arr, ImGuiColorEditFlags_Float);
					}
					else if (current_texture->type == IMAGE)
					{
						ImGui::Text("Image properties can't be edited");
					}
					else if (current_texture->type == CHECKER)
					{
						const auto current_checker = (CheckerInfo*)current_texture;
						is_edited |= ImGui::ColorEdit3("Color 1", current_checker->albedo_a.arr, ImGuiColorEditFlags_Float);
						is_edited |= ImGui::ColorEdit3("Color 2", current_checker->albedo_b.arr, ImGuiColorEditFlags_Float);
						is_edited |= ImGui::SliderFloat("Tile size", &current_checker->tile_size, 0.001f, UINT8_MAX, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
					}
					ImGui::TreePop();
				}
				ImGui::TreePop();
			}
			ImGui::PopID();

			if (is_rendering_ && is_edited)
			{
				renderer_->refresh_texture(i);
				renderer_->refresh_buffer();
				render_info_.frames_since_refresh = 0;
			}
		}
	}
}

void RtInterface::add_material()
{
	if (ImGui::CollapsingHeader("Add material"))
	{
		bool is_added = false;
		static int32_t material_type = UNKNOWN_MATERIAL;
		static int32_t selected_texture = 0;

		if (ImGui::TreeNode("Texture"))
		{
			draw_textures(world_info_.textures_.data(), (int32_t)world_info_.textures_.size(), selected_texture);
			ImGui::TreePop();
		}

		ImGui::Combo("Material type", &material_type, material_types, IM_ARRAYSIZE(material_types));
		if (material_type == UNKNOWN_MATERIAL)
		{
			ImGui::Text("Material of type Unknown_Material can't be instantiated");
		}
		else if (material_type == DIFFUSE)
		{
			if (ImGui::Button("Create material", {ImGui::GetContentRegionAvail().x, 0}))
			{
				if (is_rendering_)
					renderer_->deallocate_world();
				world_info_.add_material(new DiffuseInfo(selected_texture));
				is_added = true;
			}
		}
		else if (material_type == SPECULAR)
		{
			static float new_specular_fuzziness{0.0f};

			if (ImGui::TreeNode("Properties"))
			{
				ImGui::SliderFloat("Fuzziness", &new_specular_fuzziness, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
				draw_help("Fuzziness of material reflection");
				ImGui::TreePop();
			}

			if (ImGui::Button("Create material", {ImGui::GetContentRegionAvail().x, 0}))
			{
				if (is_rendering_)
					renderer_->deallocate_world();
				world_info_.add_material(new SpecularInfo(new_specular_fuzziness, selected_texture));
				is_added = true;
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
				if (is_rendering_)
					renderer_->deallocate_world();
				world_info_.add_material(new RefractiveInfo(new_refractive_index_of_refraction));
				is_added = true;
			}
		}
		else if (material_type == ISOTROPIC)
		{
			if (ImGui::Button("Create material", {ImGui::GetContentRegionAvail().x, 0}))
			{
				if (is_rendering_)
					renderer_->deallocate_world();
				world_info_.add_material(new IsotropicInfo(selected_texture));
				is_added = true;
			}
		}

		if (is_rendering_ && is_added)
		{
			renderer_->allocate_world();
			renderer_->refresh_buffer();
			render_info_.frames_since_refresh = 0;
		}
	}
}

void RtInterface::edit_material()
{
	if (ImGui::CollapsingHeader("Material list"))
	{
		bool is_edited = false;
		for (int32_t i = 0; i < world_info_.materials_.size(); i++)
		{
			MaterialInfo* current_material = world_info_.materials_[i];
			ImGui::PushID(i);
			if (ImGui::TreeNode("Material", "%u-%s-%s", i, material_types[current_material->type], texture_types[world_info_.textures_[current_material->texture_id]->type]))
			{
				ImGui::Text("Material type: %s", material_types[current_material->type]);

				if (ImGui::TreeNode("Texture"))
				{
					ImGui::Text("Material's texture id: %u", current_material->texture_id);
					static int32_t selected_texture = 0;
					draw_textures(world_info_.textures_.data(), (int32_t)world_info_.textures_.size(), selected_texture);

					if (ImGui::Button("Set texture", {ImGui::GetContentRegionAvail().x, 0}))
					{
						current_material->texture_id = selected_texture;
						is_edited = true;
					}
					ImGui::TreePop();
				}

				if (ImGui::TreeNode("Properties"))
				{
					if (current_material->type == UNKNOWN_MATERIAL)
					{
						ImGui::Text("Material properties can't be set for type Unknown_Material");
					}
					else if (current_material->type == DIFFUSE)
					{
						ImGui::Text("Diffuse properties can't be edited");
					}
					else if (current_material->type == SPECULAR)
					{
						const auto current_specular = (SpecularInfo*)current_material;
						is_edited |= ImGui::SliderFloat("Fuzziness", &current_specular->fuzziness, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
						draw_help("Fuzziness of material reflection");
					}
					else if (current_material->type == REFRACTIVE)
					{
						const auto current_refractive = (RefractiveInfo*)current_material;
						is_edited |= ImGui::SliderFloat("Refractive index", &current_refractive->refractive_index, 0.0f, 4.0f, "%.3f", 
							ImGuiSliderFlags_AlwaysClamp);
						draw_help("Index of refraction between air and current material");
					}
					else if (current_material->type == ISOTROPIC)
					{
						ImGui::Text("Isotropic properties can't be edited");
					}
					ImGui::TreePop();
				}
				ImGui::TreePop();
			}
			ImGui::PopID();

			if (is_rendering_ && is_edited)
			{
				renderer_->refresh_material(i);
				renderer_->refresh_buffer();
				render_info_.frames_since_refresh = 0;
			}
		}
	}
}

void RtInterface::add_object()
{
	if (ImGui::CollapsingHeader("Add object"))
	{
		bool is_added = false;
		static int32_t object_type = UNKNOWN_OBJECT;
		static int32_t selected_material = 0;

		ImGui::Combo("Object type", &object_type, object_types, IM_ARRAYSIZE(object_types));

		if (ImGui::TreeNode("Material"))
		{
			draw_materials(world_info_.textures_.data(), world_info_.materials_.data(), (int32_t)world_info_.materials_.size(), selected_material);
			ImGui::TreePop();
		}

		if (object_type == UNKNOWN_OBJECT)
		{
			ImGui::Text("Object of type Unknown_Object can't be instantiated");
		}
		else if (object_type == SPHERE)
		{
			static Float3 new_sphere_center{0.0f, 0.0f, 0.0f};
			static float new_sphere_radius{1.0f};

			if (ImGui::TreeNode("Properties"))
			{
				ImGui::SliderFloat3("Center", new_sphere_center.arr, -UINT16_MAX, UINT16_MAX,"%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
				ImGui::SliderFloat("Radius", &new_sphere_radius, -UINT8_MAX, UINT8_MAX, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
				draw_help("Radius of sphere can be negative for refractive spheres to make sphere hollow");
				ImGui::TreePop();
			}

			if (ImGui::Button("Create object", {ImGui::GetContentRegionAvail().x, 0}))
			{
				if (is_rendering_)
					renderer_->deallocate_world();
				world_info_.add_object(new SphereInfo(new_sphere_center.str, new_sphere_radius, selected_material));
				is_added = true;
			}
		}
		else if (object_type == TRIANGLE)
		{
			static Float3 new_triangle_v0{-1.0f, 0.0f, 0.0f};
			static Float3 new_triangle_v1{0.0f, 1.0f, 0.0f};
			static Float3 new_triangle_v2{1.0f, 0.0f, 0.0f};

			if (ImGui::TreeNode("Properties"))
			{
				ImGui::SliderFloat3("Vertex 0", new_triangle_v0.arr, -UINT16_MAX, UINT16_MAX, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
				ImGui::SliderFloat3("Vertex 1", new_triangle_v1.arr, -UINT16_MAX, UINT16_MAX, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
				ImGui::SliderFloat3("Vertex 2", new_triangle_v2.arr, -UINT16_MAX, UINT16_MAX, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
				ImGui::TreePop();
			}

			if (ImGui::Button("Create object", {ImGui::GetContentRegionAvail().x, 0}))
			{
				if (is_rendering_)
					renderer_->deallocate_world();
				const float3 normal = cross(new_triangle_v1.str - new_triangle_v0.str, new_triangle_v2.str - new_triangle_v0.str);
				world_info_.add_object(new TriangleInfo(new_triangle_v0.str, new_triangle_v1.str, new_triangle_v2.str, selected_material, normal, make_float2(0.0f, 0.0f), make_float2(1.0f, 1.0f)));
				is_added = true;
			}
		}
		else if (object_type == PLANE)
		{
			static Float3 new_plane_normal{0.0f, -1.0f, 0.0f};
			static float new_plane_offset{0.0f};

			if (ImGui::TreeNode("Properties"))
			{
				ImGui::SliderFloat3("Normal", new_plane_normal.arr, -1.0f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
				ImGui::SliderFloat("Offset", &new_plane_offset, -UINT16_MAX, UINT16_MAX, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
				draw_help("Displacement from center position along normal");
				ImGui::TreePop();
			}

			if (ImGui::Button("Create object", {ImGui::GetContentRegionAvail().x, 0}))
			{
				if (is_rendering_)
					renderer_->deallocate_world();
				world_info_.add_object(new PlaneInfo(new_plane_normal.str, new_plane_offset, selected_material));
				is_added = true;
			}
		}
		else if (object_type == VOLUMETRIC_SPHERE)
		{
			static Float3 new_volumetric_sphere_center{0.0f, 0.0f, 0.0f};
			static float new_volumetric_sphere_radius{1.0f};
			static float new_volumetric_sphere_density{0.1f};

			if (ImGui::TreeNode("Properties"))
			{
				ImGui::SliderFloat3("Center", new_volumetric_sphere_center.arr, -UINT16_MAX, UINT16_MAX,"%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
				ImGui::SliderFloat("Radius", &new_volumetric_sphere_radius, 0.0f, UINT8_MAX, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
				ImGui::SliderFloat("Density", &new_volumetric_sphere_density, 0.0f, UINT8_MAX, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
				draw_help("Density of volume contained within sphere");
				ImGui::TreePop();
			}

			if (ImGui::Button("Create object", {ImGui::GetContentRegionAvail().x, 0}))
			{
				if (is_rendering_)
					renderer_->deallocate_world();
				world_info_.add_object(new VolumetricSphereInfo(new_volumetric_sphere_center.str, new_volumetric_sphere_radius, new_volumetric_sphere_density, selected_material));
				is_added = true;
			}
		}
		else if (object_type == CYLINDER)
		{
			static Float3 new_cylinder_extreme_a{0.0f, 0.5f, 0.0f};
			static Float3 new_cylinder_extreme_b{0.0f, -0.5f, 0.0f};
			static Float3 new_cylinder_center{0.0f, 0.0f, 0.0f};
			static float new_cylinder_radius{0.5f};

			if (ImGui::TreeNode("Properties"))
			{
				ImGui::SliderFloat3("Extreme 1", new_cylinder_extreme_a.arr, -UINT16_MAX, UINT16_MAX, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
				ImGui::SliderFloat3("Extreme 2", new_cylinder_extreme_b.arr, -UINT16_MAX, UINT16_MAX, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
				ImGui::SliderFloat3("Center", new_cylinder_center.arr, -UINT16_MAX, UINT16_MAX, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
				ImGui::SliderFloat("Radius", &new_cylinder_radius, 0.0f, UINT8_MAX, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
				ImGui::TreePop();
			}

			if (ImGui::Button("Create object", {ImGui::GetContentRegionAvail().x, 0}))
			{
				if (is_rendering_)
					renderer_->deallocate_world();
				world_info_.add_object(new CylinderInfo(new_cylinder_extreme_a.str, new_cylinder_extreme_b.str, new_cylinder_center.str, new_cylinder_radius, selected_material));
				is_added = true;
			}
		}
		else if (object_type == CONE)
		{
			static Float3 new_cone_extreme_a{0.0f, 0.5f, 0.0f};
			static Float3 new_cone_extreme_b{0.0f, -0.5f, 0.0f};
			static Float3 new_cone_center{0.0f, 0.0f, 0.0f};
			static float new_cone_radius_a{0.1f};
			static float new_cone_radius_b{0.5f};

			if (ImGui::TreeNode("Properties"))
			{
				ImGui::SliderFloat3("Extreme 1", new_cone_extreme_a.arr, -UINT16_MAX, UINT16_MAX, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
				ImGui::SliderFloat3("Extreme 2", new_cone_extreme_b.arr, -UINT16_MAX, UINT16_MAX, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
				ImGui::SliderFloat3("Center", new_cone_center.arr, -UINT16_MAX, UINT16_MAX, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
				ImGui::SliderFloat("Radius 1", &new_cone_radius_a, 0.0f, UINT8_MAX, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
				ImGui::SliderFloat("Radius 2", &new_cone_radius_b, 0.0f, UINT8_MAX, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
				ImGui::TreePop();
			}

			if (ImGui::Button("Create object", {ImGui::GetContentRegionAvail().x, 0}))
			{
				if (is_rendering_)
					renderer_->deallocate_world();
				world_info_.add_object(new ConeInfo(new_cone_extreme_a.str, new_cone_extreme_b.str, new_cone_center.str, new_cone_radius_a, new_cone_radius_b, selected_material));
				is_added = true;
			}
		}
		else if (object_type == TORUS)
		{
			static Float3 new_torus_center{0.0f, 0.0f, 0.0f};
			static float new_torus_radius_a{0.5f};
			static float new_torus_radius_b{0.1f};

			if (ImGui::TreeNode("Properties"))
			{
				ImGui::SliderFloat3("Center", new_torus_center.arr, -UINT16_MAX, UINT16_MAX, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
				ImGui::SliderFloat("Radius 1", &new_torus_radius_a, 0.0f, UINT8_MAX, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
				ImGui::SliderFloat("Radius 2", &new_torus_radius_b, 0.0f, UINT8_MAX, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
				ImGui::TreePop();
			}

			if (ImGui::Button("Create object", {ImGui::GetContentRegionAvail().x, 0}))
			{
				if (is_rendering_)
					renderer_->deallocate_world();
				world_info_.add_object(new TorusInfo(new_torus_center.str, new_torus_radius_a, new_torus_radius_b, selected_material));
				is_added = true;
			}
		}
		else if (object_type == MODEL)
		{
			static std::filesystem::path selected_file;

			ImGui::Text("Choose 3D Model");
			draw_help("Choose .obj file containing 3D model to load. New files can be added to \"assets/obj\" folder");

			draw_files(selected_file, "3D Models", "assets/obj/");

			if (ImGui::Button("Create object", {ImGui::GetContentRegionAvail().x, 0}))
			{
				TriangleInfo* triangle_list = nullptr;
				uint64_t triangle_count = 0;
				world_info_.load_model(selected_file.u8string(), selected_material, triangle_list, triangle_count);

				if (triangle_count == 0)
					ImGui::OpenPopup("3D model loading failed");
				else
				{
					if (is_rendering_)
						renderer_->deallocate_world();
					world_info_.add_object(new ModelInfo(triangle_list, triangle_count, selected_material));
					is_added = true;
				}
			}

			draw_modal("3D model loading failed", "This file can't be loaded as 3D obj model");
		}

		if (is_rendering_ && is_added)
		{
			renderer_->allocate_world();
			renderer_->refresh_buffer();
			render_info_.frames_since_refresh = 0;
		}
	}
}

void RtInterface::edit_object()
{
	if (ImGui::CollapsingHeader("Object list"))
	{
		bool is_edited = false, is_deleted = false;
		for (int32_t i = 0; i < world_info_.objects_.size(); i++)
		{
			ObjectInfo* current_object = world_info_.objects_[i];

			ImGui::PushID(i);
			if (ImGui::TreeNode("Object", "%u-%s-%s-%s", i, object_types[current_object->type], material_types[world_info_.materials_[current_object->material_id]->type], texture_types[world_info_.textures_[world_info_.materials_[current_object->material_id]->texture_id]->type]))
			{
				ImGui::Text("Object type: %s", object_types[current_object->type]);

				if (current_object->type == MODEL)
				{
					ImGui::Text("Triangle count: %llu", ((ModelInfo*)current_object)->triangle_count);
				}

				if (ImGui::TreeNode("Material"))
				{
					ImGui::Text("Object's material id: %u", current_object->material_id);
					static int32_t selected_material = 0;
					draw_materials(world_info_.textures_.data(), world_info_.materials_.data(), (int32_t)world_info_.materials_.size(), selected_material);

					if (ImGui::Button("Set material", {ImGui::GetContentRegionAvail().x, 0}))
					{
						current_object->material_id = selected_material;
						is_edited = true;
					}
					ImGui::TreePop();
				}

				if (ImGui::TreeNode("Properties"))
				{
					if (current_object->type == UNKNOWN_OBJECT)
					{
						ImGui::Text("Object properties can't be set for type Unknown_Object");
					}
					else if (current_object->type == SPHERE)
					{
						const auto current_sphere = (SphereInfo*)current_object;

						is_edited |= ImGui::SliderFloat3("Center", current_sphere->center.arr, -UINT8_MAX, UINT8_MAX,"%.3f",
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
						is_edited |= ImGui::SliderFloat("Radius", &current_sphere->radius, -UINT16_MAX, UINT16_MAX, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
						draw_help("Radius of sphere can be negative for refractive spheres to make sphere hollow");
					}
					else if (current_object->type == TRIANGLE)
					{
						const auto current_triangle = (TriangleInfo*)current_object;

						is_edited |= ImGui::SliderFloat3("Vertex 0", current_triangle->v0.arr, -UINT16_MAX, UINT16_MAX, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
						is_edited |= ImGui::SliderFloat3("Vertex 1", current_triangle->v1.arr, -UINT16_MAX, UINT16_MAX, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
						is_edited |= ImGui::SliderFloat3("Vertex 2", current_triangle->v2.arr, -UINT16_MAX, UINT16_MAX, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
					}
					else if (current_object->type == PLANE)
					{
						const auto current_plane = (PlaneInfo*)current_object;

						is_edited |= ImGui::SliderFloat3("Normal", current_plane->normal.arr, -1.0f, 1.0f, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
						is_edited |= ImGui::SliderFloat("Offset", &current_plane->offset, -UINT16_MAX, UINT16_MAX, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
						draw_help("Displacement from center position along normal");
					}
					else if (current_object->type == VOLUMETRIC_SPHERE)
					{
						const auto current_volumetric_sphere = (VolumetricSphereInfo*)current_object;

						is_edited |= ImGui::SliderFloat3("Normal", current_volumetric_sphere->boundary.center.arr, -UINT16_MAX, UINT16_MAX, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
						is_edited |= ImGui::SliderFloat("Radius", &current_volumetric_sphere->boundary.radius, 0.0f, UINT8_MAX, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
						is_edited |= ImGui::SliderFloat("Density", &current_volumetric_sphere->density, 0.0f, UINT8_MAX, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
						draw_help("Density of volume contained within sphere");
					}
					else if (current_object->type == CYLINDER)
					{
						const auto current_cylinder = (CylinderInfo*)current_object;

						is_edited |= ImGui::SliderFloat3("Extreme 1", current_cylinder->extreme_a.arr, -UINT8_MAX, UINT8_MAX, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
						is_edited |= ImGui::SliderFloat3("Extreme 2", current_cylinder->extreme_b.arr, -UINT8_MAX, UINT8_MAX, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
						is_edited |= ImGui::SliderFloat3("Center", current_cylinder->center.arr, -UINT16_MAX, UINT16_MAX, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
						is_edited |= ImGui::SliderFloat("Radius", &current_cylinder->radius, 0.0f, UINT8_MAX, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
					}
					else if (current_object->type == CONE)
					{
						const auto current_cone = (ConeInfo*)current_object;

						is_edited |= ImGui::SliderFloat3("Extreme 1", current_cone->extreme_a.arr, -UINT8_MAX, UINT8_MAX, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
						is_edited |= ImGui::SliderFloat3("Extreme 2", current_cone->extreme_b.arr, -UINT8_MAX, UINT8_MAX, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
						is_edited |= ImGui::SliderFloat3("Center", current_cone->center.arr, -UINT16_MAX, UINT16_MAX, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
						is_edited |= ImGui::SliderFloat("Radius 1", &current_cone->radius_a, 0.0f, UINT8_MAX, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
						is_edited |= ImGui::SliderFloat("Radius 2", &current_cone->radius_b, 0.0f, UINT8_MAX, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
					}
					else if (current_object->type == TORUS)
					{
						const auto current_torus = (TorusInfo*)current_object;

						is_edited |= ImGui::SliderFloat3("Center", current_torus->center.arr, -UINT16_MAX, UINT16_MAX, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
						is_edited |= ImGui::SliderFloat("Radius 1", &current_torus->radius_a, 0.0f, UINT8_MAX, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
						is_edited |= ImGui::SliderFloat("Radius 2", &current_torus->radius_b, 0.0f, UINT8_MAX, "%.3f", 
							ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
					}
					else if (current_object->type == MODEL)
					{
						const auto current_model = (ModelInfo*)current_object;

						is_edited |= ImGui::SliderFloat3("Translation", current_model->translation.arr, -UINT16_MAX, UINT16_MAX, "%.3f", 
								ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_Logarithmic);
						is_edited |= ImGui::SliderFloat3("Scale", current_model->scale.arr, 0.001f, UINT8_MAX, "%.3f", 
								ImGuiSliderFlags_AlwaysClamp);
						is_edited |= ImGui::SliderAngle("Rotation x", &current_model->rotation.arr[0], 0.0f, 360.0f, "%.3f", 
								ImGuiSliderFlags_AlwaysClamp);
						is_edited |= ImGui::SliderAngle("Rotation y", &current_model->rotation.arr[1], 0.0f, 360.0f, "%.3f", 
								ImGuiSliderFlags_AlwaysClamp);
						is_edited |= ImGui::SliderAngle("Rotation z", &current_model->rotation.arr[2], 0.0f, 360.0f, "%.3f", 
								ImGuiSliderFlags_AlwaysClamp);
					}
					ImGui::TreePop();
				}

				if (ImGui::Button("Delete object", {ImGui::GetContentRegionAvail().x, 0}))
				{
					if (is_rendering_)
						renderer_->deallocate_world();
					world_info_.remove_object(i);
					is_deleted = true;
				}

				ImGui::TreePop();
			}
			ImGui::PopID();

			if (is_rendering_)
			{
				if (is_edited || is_deleted)
				{
					if (is_edited)
						renderer_->refresh_object(i);
					else if (is_deleted)
						renderer_->allocate_world();

					renderer_->refresh_buffer();
					render_info_.frames_since_refresh = 0;
				}
			}
		}
	}
}

void RtInterface::edit_sky()
{
	if (ImGui::CollapsingHeader("Environment"))
	{
		bool hdr_changed = false, exposure_changed = false, sky_changed = false;

		ImGui::Text("Set sky properties");

		if (ImGui::TreeNode("Sky properties"))
		{
			static float turbidity{2.5f};
			static float ground_albedo[3]{0.5f, 0.5f, 0.5f};
			static float elevation{1.25f};

			sky_changed |= ImGui::ColorEdit3("Ground albedo", ground_albedo, ImGuiColorEditFlags_Float);
			sky_changed |= ImGui::SliderFloat("Atmospheric turbidity", &turbidity, 1.0f, 8.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
			sky_changed |= ImGui::SliderAngle("Solar elevation", &elevation, 0.0f, 90.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);

			if (sky_changed)
				sky_info_.create_sky(turbidity, ground_albedo[0], ground_albedo[1], ground_albedo[2], elevation);

			ImGui::TreePop();
		}
		if (ImGui::TreeNode("HDR properties"))
		{
			static std::filesystem::path selected_file;

			draw_help("Choose file containing HDR image for environment map creation. New files can be added to \"assets/hdr\" folder");

			draw_files(selected_file, "HDR Files", "assets/hdr/");

			exposure_changed |= ImGui::SliderFloat("Exposure", &sky_info_.hdr_exposure, 0.0f, 16.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
			draw_help("Adjust brightness of HDR environment map");

			if (ImGui::Button("Set HDR", {ImGui::GetContentRegionAvail().x, 0}))
			{
				if (!sky_info_.check_hdr(selected_file.u8string().c_str()))
				{
					ImGui::OpenPopup("HDR loading failed");
				}
				else
				{
					sky_info_.load_hdr(selected_file.u8string().c_str());
					hdr_changed = true;
				}
			}

			draw_modal("HDR loading failed", "This file can't be loaded as HDR image");

			if (ImGui::Button("Clear HDR", {ImGui::GetContentRegionAvail().x, 0}))
			{
				sky_info_.clear_hdr();
				hdr_changed = true;
			}

			ImGui::TreePop();
		}

		if (!is_rendering_)
			return;
		
		if (exposure_changed || hdr_changed || sky_changed)
		{
			if (hdr_changed)
				renderer_->recreate_sky();

			renderer_->refresh_buffer();
			render_info_.frames_since_refresh = 0;
		}
	}
}

void RtInterface::save_image() const
{
	if (ImGui::CollapsingHeader("Save image"))
	{
		static const char* image_formats[]{"HDR", "PNG", "BMP", "TGA", "JPG"};
		static int32_t format = 0;
		static char buffer[128];
		ImGui::Combo("Image format", &format, image_formats, IM_ARRAYSIZE(image_formats));
		ImGui::InputText("Filename", buffer, sizeof buffer);

		if (ImGui::Button("Save", {ImGui::GetContentRegionAvail().x, 0}))
		{
			if (image_data_)
			{
				const auto output_path = std::filesystem::path("output") / buffer;

				if (format == 0)
					stbi_write_hdr((output_path.u8string() + ".hdr").c_str(), (int32_t)render_info_.width, (int32_t)render_info_.height, 4, image_data_);
				else
				{
					const uint32_t image_size = 4 * render_info_.width * render_info_.height;
					const auto data = new uint8_t[image_size];
					for (uint32_t i = 0; i <= image_size; i++)
						data[i] = (uint8_t)(clamp(image_data_[i], 0.0f, 1.0f) * 255.99f);

					if (format == 1)
						stbi_write_png((output_path.u8string() + ".png").c_str(), (int32_t)render_info_.width, (int32_t)render_info_.height, 4, data, 4 * render_info_.width);
					else if (format == 2)
						stbi_write_bmp((output_path.u8string() + ".bmp").c_str(), (int32_t)render_info_.width, (int32_t)render_info_.height, 4, data);
					else if (format == 3)
						stbi_write_tga((output_path.u8string() + ".tga").c_str(), (int32_t)render_info_.width, (int32_t)render_info_.height, 4, data);
					else if (format == 4)
						stbi_write_jpg((output_path.u8string() + ".jpg").c_str(), (int32_t)render_info_.width, (int32_t)render_info_.height, 4, data, 90);

					delete[] data;
				}
			}
			else
				ImGui::OpenPopup("Saving failed");
		}

		draw_modal("Saving failed", "Image was not rendered yet");
	}
}