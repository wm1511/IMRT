#pragma once
#include "MaterialInfo.hpp"
#include "ObjectInfo.hpp"

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

struct WorldInfo
{
	int32_t object_data_count{0}, material_data_count{0}, object_count{0}, material_count{0}, object_capacity{0}, material_capacity{0};
	ObjectInfo** object_data = nullptr;
	MaterialInfo** material_data = nullptr;

	WorldInfo()
	{
		material_data_count = 4;
		object_data_count = 4;
		material_count = 4;
		object_count = 4;
		object_capacity = 4;
		material_capacity = 4;
		material_data = new MaterialInfo*[material_count];
		object_data = new ObjectInfo*[object_count];
		material_data[0] = new DiffuseInfo({0.5f, 0.5f, 0.5f});
		material_data[1] = new RefractiveInfo(1.5f);
		material_data[2] = new SpecularInfo({0.5f, 0.5f, 0.5f}, 0.1f);
		material_data[3] = new DiffuseInfo({0.2f, 0.2f, 0.8f});
		object_data[0] = new SphereInfo({1.0f, 0.0f, -1.0f}, 0.5f, 0);
		object_data[1] = new SphereInfo({0.0f, 0.0f, -1.0f}, 0.5f, 1);
		object_data[2] = new SphereInfo({-1.0f, 0.0f, -1.0f}, 0.5f, 2);
		object_data[3] = new SphereInfo({0.0f, -100.5f, -1.0f}, 100.0f, 3);
	}

	~WorldInfo()
	{
		for (int32_t i = 0; i < object_data_count; i++)
			delete object_data[i];
		delete[] object_data;

		for (int32_t i = 0; i < material_data_count; i++)
			delete material_data[i];
		delete[] material_data;
	}

	void add_object(ObjectInfo* new_object)
	{
		extend_array(object_data, object_data_count, object_capacity);
		object_data[object_data_count] = new_object;
		object_data_count++;
	}

	void add_material(MaterialInfo* new_material)
	{
		extend_array(material_data, material_data_count, material_capacity);
		material_data[material_data_count] = new_material;
		material_data_count++;
	}

	void remove_object(const int32_t object_index)
	{
		delete object_data[object_index];
		if (object_count > 1 && object_data[object_count - 1] != object_data[object_index])
			object_data[object_index] = object_data[object_count - 1];
		object_count--;
		object_data_count--;
	}
};