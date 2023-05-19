#include "stdafx.h"
#include "WorldInfo.hpp"

WorldInfo::WorldInfo()
{
	textures_[0] = Texture(Checker(make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f), 0.05f));
	texture_names_[0] = "DemoChecker";
	textures_[1] = Texture(Solid(make_float3(0.5f, 0.5f, 0.5f)));
	texture_names_[1] = "DemoGray";
	textures_[2] = Texture(Solid(make_float3(0.2f, 0.2f, 0.8f)));
	texture_names_[2] = "DemoBlue";
	materials_[0] = Material(Diffuse());
	material_names_[0] = "DemoDiffuse";
	materials_[1] = Material(Refractive(1.5f));
	material_names_[1] = "DemoGlass";
	materials_[2] = Material(Specular(0.1f));
	material_names_[2] = "DemoFuzzySpecular";
	objects_[0] = Object(Sphere(make_float3(1.0f, 0.0f, -1.0f), 0.5f), 0, 0);
	object_names_[0] = "DemoSphereLeft";
	objects_[1] = Object(Sphere(make_float3(0.0f, 0.0f, -1.0f), 0.5f), 2, 1);
	object_names_[1] = "DemoSphereCentral";
	objects_[2] = Object(Sphere(make_float3(-1.0f, 0.0f, -1.0f), 0.5f), 1, 2);
	object_names_[2] = "DemoSphereRight";
	objects_[3] = Object(Sphere(make_float3(0.0f, -100.5f, -1.0f), 100.0f), 2, 0);
	object_names_[3] = "DemoSphereBottom";
}

WorldInfo::~WorldInfo()
{
	for (const auto& object : objects_)
	{
		if (object.type == ObjectType::MODEL)
		{
			delete[] object.model.h_vertices;
			delete[] object.model.h_indices;
		}
	}

	for (const auto& texture : textures_)
	{
		if (texture.type == TextureType::IMAGE)
			free(texture.image.h_data);
	}
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
				indices.push_back(static_cast<uint32_t>(indices.size()));
			}
			index_offset += 3;
		}
	}
}

void WorldInfo::remove_object(const int32_t object_index)
{
	objects_.erase(objects_.begin() + object_index);
	object_names_.erase(object_names_.begin() + object_index);
}
