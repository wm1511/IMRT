#include "stdafx.h"
#include "WorldInfo.hpp"
#include "../common/Math.cuh"

#include "tiny_obj_loader.h"

WorldInfo::WorldInfo()
{
	materials_[0] = new DiffuseInfo({0.5f, 0.5f, 0.5f});
	materials_[1] = new RefractiveInfo(1.5f);
	materials_[2] = new SpecularInfo({0.5f, 0.5f, 0.5f}, 0.1f);
	materials_[3] = new DiffuseInfo({0.2f, 0.2f, 0.8f});
	objects_[0] = new SphereInfo({1.0f, 0.0f, -1.0f}, 0.5f, 0);
	objects_[1] = new SphereInfo({0.0f, 0.0f, -1.0f}, 0.5f, 1);
	objects_[2] = new SphereInfo({-1.0f, 0.0f, -1.0f}, 0.5f, 2);
	objects_[3] = new SphereInfo({0.0f, -100.5f, -1.0f}, 100.0f, 3);
}

WorldInfo::~WorldInfo()
{
	for (uint64_t i = 0; i < objects_.size(); i++)
		delete objects_[i];

	for (uint64_t i = 0; i < materials_.size(); i++)
		delete materials_[i];
}

void WorldInfo::load_model(const std::string& model_path, const int32_t material_id, TriangleInfo*& triangles, uint64_t& triangle_count) const
{
	tinyobj::ObjReaderConfig reader_config;
	reader_config.vertex_color = false;
	reader_config.triangulation_method = "earcut";
	reader_config.mtl_search_path = "../mtl";

	tinyobj::ObjReader reader;

	if (!reader.ParseFromFile(model_path, reader_config))
		return;

	auto& attrib = reader.GetAttrib();
	auto& shapes = reader.GetShapes();

	for (uint64_t s = 0; s < shapes.size(); s++) 
		triangle_count += shapes[s].mesh.num_face_vertices.size();

	uint64_t triangle_index = 0;
	triangles = new TriangleInfo[triangle_count];

	for (uint64_t s = 0; s < shapes.size(); s++) 
	{
		for (uint64_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) 
		{
			bool has_normals = false, has_uvs = false;
			float3 vertices[3]{};
			float3 normals[3]{};
			float2 uvs[3]{};

		    for (size_t v = 0; v < 3; v++) 
			{
				const tinyobj::index_t idx = shapes[s].mesh.indices[3 * f + v];

				vertices[v] = make_float3(
					attrib.vertices[3 * idx.vertex_index],
					attrib.vertices[3 * idx.vertex_index + 1],
					attrib.vertices[3 * idx.vertex_index + 2]);

				if (idx.normal_index >= 0) 
				{
					normals[v] = make_float3(
						attrib.normals[3 * idx.normal_index],
						attrib.normals[3 * idx.normal_index + 1],
						attrib.normals[3 * idx.normal_index + 2]);
					has_normals = true;
				}

				if (idx.texcoord_index >= 0) 
				{
					uvs[v] = make_float2(
						attrib.texcoords[ 2 * idx.texcoord_index],
						attrib.texcoords[ 2 * idx.texcoord_index + 1]);
					has_uvs = true;
				}
		    }

			float3 normal = {0.0f, 0.0f, 0.0f};
			if (has_normals)
				normal = (normals[0] + normals[1] + normals[2]) / 3;

			float2 min_uv{0.0f, 0.0f}, max_uv{1.0f, 1.0f};
			if (has_uvs)
			{
				min_uv = fminf(uvs[0], uvs[1], uvs[2]);
				max_uv = fmaxf(uvs[0], uvs[1], uvs[2]);
			}

			triangles[triangle_index++] = std::move(TriangleInfo(vertices[0], vertices[1], vertices[2], material_id, normal, min_uv, max_uv));
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

void WorldInfo::remove_object(const int32_t object_index)
{
	if (*(objects_.end() - 1) != objects_[object_index])
		objects_.erase(objects_.begin() + object_index);
	else
		objects_.pop_back();
}
