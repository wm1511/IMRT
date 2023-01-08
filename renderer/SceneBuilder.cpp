#include "SceneBuilder.hpp"

#include "Sphere.hpp"
#include "Plane.hpp"
#include "Triangle.hpp"
#include "Diffuse.hpp"
#include "Specular.hpp"
#include "Refractive.hpp"

Scene SceneBuilder::MakeCornellBox()
{
    Scene scene;

    const auto specular = std::make_shared<Specular>(glm::dvec3(1.0, 1.0, 1.0), 0.0);
    const auto refractive = std::make_shared<Refractive>(glm::dvec3(1.0, 1.0, 1.0), 1.5);
    const auto blue = std::make_shared<Diffuse>(glm::dvec3(0.4, 0.4, 1.0));
    const auto light = std::make_shared<Diffuse>(glm::dvec3(0.6, 0.2, 1.0), 10000.0);
    const auto gray = std::make_shared<Diffuse>(glm::dvec3(0.6, 0.6, 0.6));
    const auto red = std::make_shared<Diffuse>(glm::dvec3(1.0, 0.2, 0.2));
    const auto green = std::make_shared<Diffuse>(glm::dvec3(0.2, 1.0, 0.2));

	scene.add(std::make_shared<Sphere>(glm::dvec3(-0.75, -1.45, -4.4), 1.05), specular);
    scene.add(std::make_shared<Sphere>(glm::dvec3(2.0, -2.05, -3.7), 0.5), refractive);
    scene.add(std::make_shared<Sphere>(glm::dvec3(-1.75, -1.95, -3.1), 0.6), blue);
    scene.add(std::make_shared<Sphere>(glm::dvec3(0, 1.9, -3), 0.5), light);

    scene.add(std::make_shared<Plane>(glm::dvec3(1, 0, 0), 2.75), red);
    scene.add(std::make_shared<Plane>(glm::dvec3(-1, 0, 0), 2.75), green);
    scene.add(std::make_shared<Plane>(glm::dvec3(0, 1, 0), 2.5), gray);
    scene.add(std::make_shared<Plane>(glm::dvec3(0, -1, 0), 3.0), gray);
    scene.add(std::make_shared<Plane>(glm::dvec3(0, 0, 1), 5.5), gray);
    scene.add(std::make_shared<Plane>(glm::dvec3(0, 0, -1), 0.5), gray);

    return scene;
}

Scene SceneBuilder::MakeUkraine()
{
    Scene scene;

    const auto blue = std::make_shared<Diffuse>(glm::dvec3(0.0, 0.0, 1.0));
    const auto yellow = std::make_shared<Diffuse>(glm::dvec3(1.0, 1.0, 0.0));
    const auto gray = std::make_shared<Diffuse>(glm::dvec3(0.6, 0.6, 0.6));
    const auto refractive = std::make_shared<Refractive>(glm::dvec3(1.0, 1.0, 1.0), 1.5);
    const auto specular = std::make_shared<Specular>(glm::dvec3(1.0, 1.0, 1.0), 0.5);
    const auto light = std::make_shared<Diffuse>(glm::dvec3(1.0, 1.0, 1.0), 5000.0);

    scene.add(std::make_shared<Sphere>(glm::dvec3(0, 5, -1), 2.0), light);
    scene.add(std::make_shared<Sphere>(glm::dvec3(0, -2, -3), 1.0), refractive);
    scene.add(std::make_shared<Sphere>(glm::dvec3(0, 0, -4), 1.0), specular);

    scene.add(std::make_shared<Plane>(glm::dvec3(0, -sqrt(3)/2, 0.5), 5.5), blue);
    scene.add(std::make_shared<Plane>(glm::dvec3(0, 1, 0), 2.5), yellow);
    scene.add(std::make_shared<Plane>(glm::dvec3(1, 0, 0), 3), gray);
    scene.add(std::make_shared<Plane>(glm::dvec3(-1, 0, 0), 3), gray);
    scene.add(std::make_shared<Plane>(glm::dvec3(0, 0, -1), 0.5), gray);

    return scene;
}

Scene SceneBuilder::MakeChoinka()
{
    Scene scene;

    const auto green = std::make_shared<Diffuse>(glm::dvec3(0.2, 1.0, 0.2));
    const auto gray = std::make_shared<Diffuse>(glm::dvec3(0.6, 0.6, 0.6));
    const auto light = std::make_shared<Diffuse>(glm::dvec3(0.6, 0.2, 1.0), 10000.0);
    const auto refractive = std::make_shared<Refractive>(glm::dvec3(1.0, 1.0, 1.0), 1.5);
    const auto specular = std::make_shared<Specular>(glm::dvec3(1.0, 1.0, 1.0), 0.0);

    scene.add(std::make_shared<Sphere>(glm::dvec3(0, 2.25, -3), 0.5), light);
    scene.add(std::make_shared<Sphere>(glm::dvec3(0.15, 0.95, -3), 0.1), refractive);
    scene.add(std::make_shared<Sphere>(glm::dvec3(-0.05, 0.1, -3), 0.1), refractive);
    scene.add(std::make_shared<Sphere>(glm::dvec3(0.35, -1.15, -3), 0.1), refractive);
    scene.add(std::make_shared<Sphere>(glm::dvec3(0.15, -1.9, -3), 0.1), refractive);
	scene.add(std::make_shared<Sphere>(glm::dvec3(-0.2, 1.2, -3), 0.1), specular);
	scene.add(std::make_shared<Sphere>(glm::dvec3(0.2, -0.15, -3), 0.1), specular);
    scene.add(std::make_shared<Sphere>(glm::dvec3(-0.25, -0.90, -3), 0.1), specular);
    scene.add(std::make_shared<Sphere>(glm::dvec3(-0.2, -2.3, -3), 0.1), specular);

    scene.add(std::make_shared<Plane>(glm::dvec3(1, 0, 0), 2.75), gray);
    scene.add(std::make_shared<Plane>(glm::dvec3(-1, 0, 0), 2.75), gray);
    scene.add(std::make_shared<Plane>(glm::dvec3(0, 1, 0), 2.5), gray);
    scene.add(std::make_shared<Plane>(glm::dvec3(0, -1, 0), 3.0), gray);
    scene.add(std::make_shared<Plane>(glm::dvec3(0, 0, 1), 5.5), gray);
    scene.add(std::make_shared<Plane>(glm::dvec3(0, 0, -1), 0.5), gray);

    scene.add(std::make_shared<Triangle>(glm::dvec3(-1, -2.5, -3), glm::dvec3(0, -1, -3), glm::dvec3(1, -2.5, -3)), green);
    scene.add(std::make_shared<Triangle>(glm::dvec3(-1, -1.5, -3), glm::dvec3(0, 0, -3), glm::dvec3(1, -1.5, -3)), green);
    scene.add(std::make_shared<Triangle>(glm::dvec3(-1, -0.5, -3), glm::dvec3(0, 1, -3), glm::dvec3(1, -0.5, -3)), green);
    scene.add(std::make_shared<Triangle>(glm::dvec3(-1, 0.5, -3), glm::dvec3(0, 2, -3), glm::dvec3(1, 0.5, -3)), green);

    return scene;
}