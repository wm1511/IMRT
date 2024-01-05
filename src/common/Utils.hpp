// Copyright Wiktor Merta 2023
#pragma once
#include <filesystem>
#include <cstdint>
#include <fstream>

#define CCE(val) check_result<cudaError_t>( "CUDA", (val), #val, __FILE__, __LINE__ )
#define COE(val) check_result<OptixResult>( "OPTIX", (val), #val, __FILE__, __LINE__ )

// Helper function writing potential error message
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

// Getting frame buffer and passing it to CUDA
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

// Reading CUDA/Optix program code
inline __host__ std::string read_shader(const std::string& program_name)
{
#ifdef _DEBUG
	const std::filesystem::path path = std::filesystem::current_path() / "x64" / "Debug" / program_name;
#else
	const std::filesystem::path path = std::filesystem::current_path() / "x64" / "Release" / program_name;
#endif

	std::ifstream file(path, std::ios::in | std::ios::binary);
    const uint64_t size = file_size(path);
    std::string source(size, '\0');
	
    file.read(source.data(), static_cast<int64_t>(size));
	file.close();

	return source;
}