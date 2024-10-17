#ifndef CUDNN_TEST_LAYER
#define CUDNN_TEST_LAYER
#pragma once

#include "tensor.cuh"

class Layer {
public:
    Layer() {}
    virtual ~Layer() {}

    virtual Tensor& forward(const Tensor& x, Tensor& y) = 0;
};

#endif // CUDNN_TEST_LAYER