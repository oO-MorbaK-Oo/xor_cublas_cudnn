#ifndef CUDNN_TEST_TENSOR
#define CUDNN_TEST_TENSOR
#pragma once
#include <vector>
#include "global.cuh"


// Row major order tensor

class Tensor {
public:
    Tensor() {}
    virtual ~Tensor() { free(); }

    cudaError alloc(const std::vector<size_t>& shape);
    bool      loadfp32(const char* filename);
    void      zeros();
    void      fill(const float value);
    void      fill(const Tensor& value);
    void      toDevice();
    void      toHost();
    void      free();
    void      print() const;

    const std::vector<size_t>& shape() const { return _shape; }
    size_t                     numel() const { return _numElements; }
    size_t                     ndimension() const { return _shape.size(); }
    float* host() { return _host; }
    float* device() { return _device; }
    const float* host() const { return _host; }
    const float* device() const { return _device; }
    cudnnTensorDescriptor_t desc() const { return _tensorDesc; }


private:
    std::vector<size_t> _shape;
    size_t _elementSize = 4;
    size_t _numElements = 0;
    size_t _size = 0;
    float* _host = nullptr;
    float* _device = nullptr;
    cudnnTensorDescriptor_t _tensorDesc = nullptr;
    Tensor* _grad = nullptr;
};

#endif // CUDNN_TEST_BUFFER