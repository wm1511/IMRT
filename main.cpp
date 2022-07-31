#include "renderer/Renderer.hpp"
#include "wrapper/App.hpp"

int main()
{
	imrt::AppInfo appInfo;
	appInfo.name = "Immediate Mode Ray Tracer";
	appInfo.fontSize = 22.0f;
	appInfo.width = 1920;
	appInfo.height = 1080;

	try
	{
		const auto app = new imrt::App(appInfo);
		app->setInterface<imrt::Renderer>();
		app->run();
		delete app;
	}
	catch (const std::exception& e)
	{
		fprintf(stderr, e.what());
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}