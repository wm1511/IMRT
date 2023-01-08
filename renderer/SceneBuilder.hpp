#pragma once
#include "Scene.hpp"


class SceneBuilder
{
public:
	SceneBuilder() = delete;

	[[nodiscard]] static Scene MakeCornellBox();
	[[nodiscard]] static Scene MakeUkraine();
	[[nodiscard]] static Scene MakeChoinka();

};

