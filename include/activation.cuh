#ifndef CUDNN_TEST_ACTIVATION
#define CUDNN_TEST_ACTIVATION
#pragma once

#include "Layer.cuh"

class Activation : public Layer {
public:
    Activation(cudnnActivationMode_t mode = CUDNN_ACTIVATION_SIGMOID, double coef = 1.0f, cudnnNanPropagation_t reluNanOpt = CUDNN_NOT_PROPAGATE_NAN);
    virtual ~Activation();

    Tensor& forward(const Tensor& x, Tensor& y);
private:
    cudnnActivationDescriptor_t _desc = nullptr;
};

#endif // CUDNN_TEST_ACTIVATION