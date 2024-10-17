#ifndef CUDNN_TEST_GLOBAL
#define CUDNN_TEST_GLOBAL
#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include <thrust/random.h>
#include <cudnn.h>

namespace Global {
    void initialize();
    void shutdown();

    cublasHandle_t getCublasHandle();
    cudnnHandle_t getCudnnHandle();
}

#endif // CUDNN_TEST_GLOBAL