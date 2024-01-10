# IMRT
###### Real-time CPU/CUDA/OPTIX path-tracer using ImGui
## Description
IMRT (Immediate Mode Ray Tracer) is an engine developed as my engineering thesis, based on a path-tracing technique. Main goals set for the project were maximal interactivity with user and availability of implementations for many platforms, including one offering hardware accelerated path-tracing.
## Screenshot
![image](https://github.com/wm1511/IMRT/assets/72276813/a01183ce-49d3-464c-9026-97f728343f02)
## Architecture
The application is divided into two layers implemented mainly in the object-oriented paradigm. The first one is responsible for drawing the window and interface, allows control over the interface and processes the rendered data. The second is a rendering layer responsible for rendering a 
scene defined by the user. Rendering is performed by one of 3 available engines: CPU, CUDA or OPTIX. Each of them communicates with the interface using shared data structures.
## Libraries
* [Dear ImGui](https://github.com/ocornut/imgui)
* [Vulkan](https://www.vulkan.org/)
* [GLFW](https://github.com/glfw/glfw)
* [ArHosekSkyModel](https://cgg.mff.cuni.cz/projects/SkylightModelling/)
* [stb_image, stb_image_write](https://github.com/nothings/stb)
* [tiny_obj_loader](https://github.com/tinyobjloader/tinyobjloader)
* [OpenMP](https://www.openmp.org/)
* [CUDA](https://developer.nvidia.com/cuda-toolkit)
* [OptiX](https://developer.nvidia.com/rtx/ray-tracing/optix)
## UML Class Diagram
![Klasy](https://github.com/wm1511/IMRT/assets/72276813/4f8317cb-10b9-4a2e-ba67-ed1f5370ac8e)
## Performance
The scenes in the image below were generated using available engines. The number of generated frames for each engine is 100. Not all scenes were generated using each engine due to excessive rendering time.
![Sceny](https://github.com/wm1511/IMRT/assets/72276813/1300cc6f-0980-4678-b63a-4a3f602951ec)
**Render time depending on used scene and engine**
|         | CPU Engine | CUDA Engine | OPTIX Engine |
| ------- | ---------- | ----------- | ------------ |
| Scene 1 | 2s 759ms | 142ms | 207ms |
| Scene 2 | 4s 845ms | 303ms | 517ms |
| Scene 3 | 4s 658ms | 401ms | 548ms |
| Scene 4 | 4m 43s 552ms | 19s 51ms | 662ms |
| Scene 5 | - | 30m 14s 216ms | 1s 794ms |
| Scene 6 | - | - | 7s 434ms |
## Potential development
* Texture and material system (probably based on some material definition language)
* More capable 3D model importer
* Instancing support
* CPU and CUDA BVH implementation
