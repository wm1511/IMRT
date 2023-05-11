#pragma once
#include <filesystem>
#include <cstdint>
#include <fstream>

#define CCE(val) check_result<cudaError_t>( "CUDA", (val), #val, __FILE__, __LINE__ )
#define COE(val) check_result<OptixResult>( "OPTIX", (val), #val, __FILE__, __LINE__ )

template <typename T>
__host__ void check_result(const char* library, const T result, char const* const func, const char* const file, int const line)
{
	if (result)
	{
		printf("%s error = %u at %s: %i '%s' \n", library, static_cast<int32_t>(result), file, line, func);
		cudaDeviceReset();
		abort();
	}
}

inline __host__ void* fetch_external_memory(void* memory_handle, const uint64_t memory_size)
{
	void* buffer;
	cudaExternalMemoryHandleDesc handle_desc = {};

#if defined(_WIN32)
	handle_desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
	handle_desc.handle.win32.handle = memory_handle;
#elif defined(__linux__) || defined(__APPLE__)
	handle_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
	handle_desc.handle.fd = reinterpret_cast<int>(memory_handle);
#endif

	handle_desc.size = memory_size;

	cudaExternalMemory_t external_memory = {};
	CCE(cudaImportExternalMemory(&external_memory, &handle_desc));

	cudaExternalMemoryBufferDesc buffer_desc = {};
	buffer_desc.size = memory_size;
	buffer_desc.offset = 0;

	CCE(cudaExternalMemoryGetMappedBuffer(&buffer, external_memory, &buffer_desc));
	return buffer;
}

inline __host__ std::string read_ptx(const std::string& program_name)
{
	int32_t device_number{};
	cudaGetDevice(&device_number);

	cudaDeviceProp device_prop{};
	cudaGetDeviceProperties(&device_prop, device_number);

	const std::string filename = program_name + ".compute_" + std::to_string(device_prop.major * 10 + device_prop.minor) + ".ptx";

#ifdef _DEBUG
	const std::filesystem::path path = std::filesystem::current_path() / "x64" / "Debug" / filename;
#else
	const std::filesystem::path path = std::filesystem::current_path() / "x64" / "Release" / filename;
#endif

	std::ifstream file(path, std::ios::in | std::ios::binary);
    const uint64_t size = file_size(path);
    std::string source(size, '\0');

    file.read(source.data(), static_cast<int64_t>(size));
	file.close();

	return source;
}