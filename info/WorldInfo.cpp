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

void WorldInfo::load_model(const std::string& model_path, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices) const
{
	tinyobj::ObjReaderConfig reader_config;
	reader_config.vertex_color = false;
	reader_config.triangulation_method = "earcut";

	tinyobj::ObjReader reader;

	if (!reader.ParseFromFile(model_path, reader_config))
		return;

	auto& attrib = reader.GetAttrib();
	auto& shapes = reader.GetShapes();

	for (const auto& shape : shapes)
	{
		size_t index_offset = 0;
		for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++)
		{
			for (size_t v = 0; v < 3; v++)
			{
				Vertex vertex{};
				const auto index = shape.mesh.indices[index_offset + v];

				vertex.position =
				{
					attrib.vertices[3 * index.vertex_index + 0],
					attrib.vertices[3 * index.vertex_index + 1],
					attrib.vertices[3 * index.vertex_index + 2],
				};

				if (index.normal_index >= 0)
				{
					vertex.normal =
					{
						attrib.normals[3 * index.normal_index + 0],
						attrib.normals[3 * index.normal_index + 1],
						attrib.normals[3 * index.normal_index + 2],
					};
				}

				if (index.texcoord_index >= 0)
				{
					vertex.uv =
					{
						attrib.texcoords[2 * index.texcoord_index + 0],
						attrib.texcoords[2 * index.texcoord_index + 1],
					};
				}


				vertices.push_back(vertex);
				indices.push_back(indices.size());
			}
			index_offset += 3;
		}
	}
}

void WorldInfo::remove_object(const int32_t object_index)
{
	if (*(objects_.end() - 1) != objects_[object_index])
		objects_.erase(objects_.begin() + object_index);
	else
		objects_.pop_back();
}
