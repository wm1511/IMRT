// Copyright (c) 2022, Wiktor Merta
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "Scene.hpp"

#include "Sphere.hpp"
#include "Square.hpp"
#include "Diffuse.hpp"
#include "Specular.hpp"
#include "Refractive.hpp"

Intersection Scene::intersect(const Ray& ray) const
{
    Intersection closest;

    for (auto& object : mObjects)
    {
	    const float t = object->intersect(ray);
        if (t > std::numeric_limits<float>::epsilon() && t < closest.getT())
        {
	        closest.setObject(object);
            closest.setT(t);
        }
    }
    return closest;
}

Scene Scene::makeCornellBox()
{
    Scene scene;
    scene.mSceneInfo.samplesPerPixel = 8;

    const auto specular = std::make_shared<Specular>(glm::vec3(4, 8, 4));
    const auto refractive = std::make_shared<Refractive>(glm::vec3(10, 10, 1), 1.5f);
    const auto blue = std::make_shared<Diffuse>(glm::vec3(4, 4, 12));
    const auto light = std::make_shared<Diffuse>(glm::vec3(0, 0, 0), 10000.0f);
    const auto gray = std::make_shared<Diffuse>(glm::vec3(6, 6, 6));
    const auto red = std::make_shared<Diffuse>(glm::vec3(10, 2, 2));
    const auto green = std::make_shared<Diffuse>(glm::vec3(2, 10, 2));

	scene.add(std::make_shared<Sphere>(glm::vec3(-0.75, -1.45, -4.4), 1.05f), specular);
    scene.add(std::make_shared<Sphere>(glm::vec3(2.0, -2.05, -3.7), 0.5f), refractive);
    scene.add(std::make_shared<Sphere>(glm::vec3(-1.75, -1.95, -3.1), 0.6f), blue);
    scene.add(std::make_shared<Sphere>(glm::vec3(0, 1.9, -3), 0.5f), light);

    scene.add(std::make_shared<Square>(glm::vec3(1, 0, 0), 2.75f), red);
    scene.add(std::make_shared<Square>(glm::vec3(-1, 0, 0), 2.75f), green);
    scene.add(std::make_shared<Square>(glm::vec3(0, 1, 0), 2.5f), gray);
    scene.add(std::make_shared<Square>(glm::vec3(0, -1, 0), 3.0f), gray);
    scene.add(std::make_shared<Square>(glm::vec3(0, 0, 1), 5.5f), gray);
    scene.add(std::make_shared<Square>(glm::vec3(0, 0, -1), 0.5f), gray);

    return scene;
}

SceneInfo Scene::getInfo() const
{
    return mSceneInfo;
}

void Scene::add(std::shared_ptr<Object> object, const std::shared_ptr<Material>& material)
{
    object->setMaterial(material);
    mObjects.push_back(std::move(object));
}
