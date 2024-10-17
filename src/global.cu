#include "global.cuh"

namespace Global {

    cublasHandle_t _h_cublas = nullptr;
    cudnnHandle_t _h_cudnn = nullptr;

    void initialize()
    {
        auto cudaStatus = cudaSetDevice(0);

        cublasCreate_v2(&_h_cublas);

        auto cudnnStatus = cudnnCreate(&_h_cudnn);
    }

    void shutdown()
    {
        cudnnDestroy(_h_cudnn);
        cublasDestroy_v2(_h_cublas);
        cudaDeviceReset();
    }

    cublasHandle_t getCublasHandle()
    {
        return _h_cublas;
    }

    cudnnHandle_t getCudnnHandle()
    {
        return _h_cudnn;
    }
}