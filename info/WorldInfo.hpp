#pragma once
#include "Texture.hpp"
#include "Material.hpp"
#include "Object.hpp"

#include <vector>
#include <string>

template <typename E>
constexpr std::underlying_type_t<E> enum_cast(E e) noexcept
{
	return static_cast<std::underlying_type_t<E>>(e);
}

class WorldInfo
{
public:
	WorldInfo();
	~WorldInfo();

	WorldInfo(const WorldInfo&) = delete;
	WorldInfo(WorldInfo&&) = delete;
	WorldInfo& operator=(const WorldInfo&) = delete;
	WorldInfo& operator=(WorldInfo&&) = delete;

	void load_model(const std::string& model_path, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices) const;

	template <typename T, typename... Args> void add_object(const std::string& name, const int32_t texture, const int32_t material, Args&&... args)
	{
		objects_.push_back(Object(T(std::forward<Args>(args)...), texture, material));
		object_names_.push_back(name);
	}

	template <typename T, typename... Args> void add_material(const std::string& name, Args&&... args)
	{
		materials_.push_back(Material(T(std::forward<Args>(args)...)));
		material_names_.push_back(name);
	}

	template <typename T, typename... Args> void add_texture(const std::string& name, Args&&... args)
	{
		textures_.push_back(Texture(T(std::forward<Args>(args)...)));
		texture_names_.push_back(name);
	}

	void remove_object(int32_t object_index);

	std::vector<Texture> textures_{3};
	std::vector<std::string> texture_names_{3};
	std::vector<Material> materials_{3};
	std::vector<std::string> material_names_{3};
	std::vector<Object> objects_{4};
	std::vector<std::string> object_names_{4};
};