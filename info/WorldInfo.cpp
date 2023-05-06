#include "stdafx.h"
#include "WorldInfo.hpp"
#include "../common/Math.cuh"

#include "tiny_obj_loader.h"

WorldInfo::WorldInfo()
{
	textures_[0] = new CheckerInfo({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, 0.05f, "DemoChecker");
	textures_[1] = new SolidInfo({0.5f, 0.5f, 0.5f}, "DemoGray");
	textures_[2] = new SolidInfo({0.2f, 0.2f, 0.8f}, "DemoBlue");
	materials_[0] = new DiffuseInfo(0, "DemoDiffuseChecker");
	materials_[1] = new RefractiveInfo(1.5f, "DemoRefractive");
	materials_[2] = new SpecularInfo(0.1f, 1, "DemoFuzzySpecular");
	materials_[3] = new DiffuseInfo(2, "DemoDiffuseBlue");
	objects_[0] = new SphereInfo({1.0f, 0.0f, -1.0f}, 0.5f, 0, "DemoCheckerSphere");
	objects_[1] = new SphereInfo({0.0f, 0.0f, -1.0f}, 0.5f, 1, "DemoGlassSphere");
	objects_[2] = new SphereInfo({-1.0f, 0.0f, -1.0f}, 0.5f, 2, "DemoSpecularSphere");
	objects_[3] = new SphereInfo({0.0f, -100.5f, -1.0f}, 100.0f, 3, "DemoBaseSphere");
}

WorldInfo::~WorldInfo()
{
	for (uint64_t i = 0; i < objects_.size(); i++)
		delete objects_[i];

	for (uint64_t i = 0; i < materials_.size(); i++)
		delete materials_[i];

	for (uint64_t i = 0; i < textures_.size(); i++)
		delete textures_[i];
}

void WorldInfo::load_model(const std::string& model_path, Vertex*& vertices, uint64_t& triangle_count) const
{
	tinyobj::ObjReaderConfig reader_config;
	reader_config.vertex_color = false;
	reader_config.triangulation_method = "earcut";

	tinyobj::ObjReader reader;

	if (!reader.ParseFromFile(model_path, reader_config))
		return;

	auto& attrib = reader.GetAttrib();
	auto& shapes = reader.GetShapes();

	for (uint64_t s = 0; s < shapes.size(); s++) 
		triangle_count += shapes[s].mesh.num_face_vertices.size();

	uint64_t vertex_index = 0;
	vertices = new Vertex[3 * triangle_count];

	for (uint64_t s = 0; s < shapes.size(); s++) 
	{
		for (uint64_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) 
		{
		    for (size_t v = 0; v < 3; v++) 
			{
				Vertex vertex{};
				const tinyobj::index_t idx = shapes[s].mesh.indices[3 * f + v];

				vertex.position = make_float3(
					attrib.vertices[3 * idx.vertex_index],
					attrib.vertices[3 * idx.vertex_index + 1],
					attrib.vertices[3 * idx.vertex_index + 2]);

				if (idx.normal_index >= 0) 
				{
					vertex.normal = make_float3(
						attrib.normals[3 * idx.normal_index],
						attrib.normals[3 * idx.normal_index + 1],
						attrib.normals[3 * idx.normal_index + 2]);
				}

				if (idx.texcoord_index >= 0) 
				{
					vertex.uv = make_float2(
						attrib.texcoords[2 * idx.texcoord_index],
						attrib.texcoords[2 * idx.texcoord_index + 1]);
				}

				vertices[vertex_index++] = std::move(vertex);
			}
		}
	}
}

void WorldInfo::add_object(ObjectInfo* new_object)
{
	objects_.push_back(new_object);
}

void WorldInfo::add_material(MaterialInfo* new_material)
{
	materials_.push_back(new_material);
}

void WorldInfo::add_texture(TextureInfo* new_texture)
{
	textures_.push_back(new_texture);
}

void WorldInfo::remove_object(const int32_t object_index)
{
	if (*(objects_.end() - 1) != objects_[object_index])
		objects_.erase(objects_.begin() + object_index);
	else
		objects_.pop_back();
}
