#include "wrapper/RtInterface.hpp"
#include "wrapper/App.hpp"

int main()
{
	AppInfo app_info;
	app_info.name = "Immediate Mode Ray Tracer";
	app_info.font_size = 22.0f;
	app_info.width = 1920;
	app_info.height = 1080;

	try
	{
		App app(app_info);
		app.SetInterface<RtInterface>();
		app.run();
	}
	catch (const std::exception& e)
	{
		fprintf(stderr, e.what());
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}