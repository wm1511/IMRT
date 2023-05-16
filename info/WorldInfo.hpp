#pragma once
#include "TextureInfo.hpp"
#include "MaterialInfo.hpp"
#include "ObjectInfo.hpp"

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

	WorldInfo(const WorldInfo&) = default;
	WorldInfo(WorldInfo&&) = default;
	WorldInfo& operator=(const WorldInfo&) = default;
	WorldInfo& operator=(WorldInfo&&) = default;

	void load_model(const std::string& model_path, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices) const;

	template <typename T, typename... Args> void add_object(Args&&... args)
	{
		objects_.push_back(new T(std::forward<Args>(args)...));
	}

	template <typename T, typename... Args> void add_material(Args&&... args)
	{
		materials_.push_back(new T(std::forward<Args>(args)...));
	}

	template <typename T, typename... Args> void add_texture(Args&&... args)
	{
		textures_.push_back(new T(std::forward<Args>(args)...));
	}

	void remove_object(int32_t object_index);

	std::vector<TextureInfo*> textures_{3};
	std::vector<MaterialInfo*> materials_{4};
	std::vector<ObjectInfo*> objects_{4};
};