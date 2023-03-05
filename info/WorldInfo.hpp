#pragma once
#include "TextureInfo.hpp"
#include "MaterialInfo.hpp"
#include "ObjectInfo.hpp"

#include <vector>
#include <string>

class WorldInfo
{
public:
	WorldInfo();
	~WorldInfo();

	WorldInfo(const WorldInfo&) = delete;
	WorldInfo(WorldInfo&&) = delete;
	WorldInfo operator=(const WorldInfo&) = delete;
	WorldInfo operator=(WorldInfo&&) = delete;

	void load_model(const std::string& model_path, int32_t material_id, TriangleInfo*& triangles, uint64_t& triangle_count) const;
	void add_object(ObjectInfo* new_object);
	void add_material(MaterialInfo* new_material);
	void add_texture(TextureInfo* new_texture);
	void remove_object(int32_t object_index);

	std::vector<TextureInfo*> textures_{3};
	std::vector<MaterialInfo*> materials_{4};
	std::vector<ObjectInfo*> objects_{4};
};