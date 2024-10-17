#ifndef CUDNN_TEST_LINEAR
#define CUDNN_TEST_LINEAR
#pragma once

#include "Layer.cuh"

// Applies an affine linear transformation to the incoming data
// y= x*transpose(A) + b

class Linear : public Layer {
public:
    Linear(const size_t inFeatures, const size_t outFeatures, const bool bias = true);
    Linear(const size_t inFeatures, const size_t outFeatures, const float* weights, const float* bias);
    Linear(const size_t inFeatures, const size_t outFeatures, const char* weightsFile, const char* biasFile);
    virtual ~Linear();

    const Tensor& weight() const { return _weight; }
    const Tensor& bias() const { return _bias; }

    Tensor& forward(const Tensor& x, Tensor& y);
private:
    size_t _inFeatures;
    size_t _outFeatures;
    bool _bBias;
    Tensor _weight;
    Tensor _bias;
};

#endif // CUDNN_TEST_LINEAR
