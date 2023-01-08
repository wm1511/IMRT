#include "CudaRenderer.cuh"

#include <cuda_runtime.h>
#include <curand.h>

CudaRenderer::CudaRenderer(const RtInfo& rt_info) : rt_info_(rt_info)
{
}

void CudaRenderer::render(uint32_t* image_data, const uint32_t width, const uint32_t height)
{
    curandGenerator_t gen;
    float *dev_data;

    cudaMalloc(reinterpret_cast<void**>(&dev_data), width * height * sizeof(float));

    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniform(gen, dev_data, width * height);

    cudaMemcpy(image_data, dev_data, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    curandDestroyGenerator(gen);
    cudaFree(dev_data);
}