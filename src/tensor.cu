#include "Tensor.cuh"
#include <numeric>
#include <functional>
#include <algorithm>
#include <assert.h>



cudaError Tensor::alloc(const std::vector<size_t>& shape)
{
    if (shape == _shape)
        return cudaSuccess;
    cudaError status = cudaErrorInvalidValue;
    free();

    if (shape.empty() || shape.size() > 5)
        return status;

    size_t numElements = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    if (numElements == 0) {
        return status;
    }

    size_t elementSize = sizeof(float);
    size_t size = numElements * elementSize;

    status = cudaMallocHost(&_host, size);
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaMalloc(&_device, size);
    if (status != cudaSuccess) {
        free();
        return status;
    }

    cudnnStatus_t dnnStatus = cudnnCreateTensorDescriptor(&_tensorDesc);
    if (dnnStatus != CUDNN_STATUS_SUCCESS) {
        free();
        return cudaErrorUnknown;
    }

    // transform size_t vector to int vector
    std::vector<int> iShape(5, 1);
    std::transform(shape.cbegin(), shape.cend(), iShape.begin(), [](const size_t s) { return (int)s; });

    // calculate tensor strides
    std::vector<int> iStride(5, 1);
    int stride = 1;
    for (int i = (int)iShape.size() - 1; i >= 0; --i) {
        iStride[i] = stride;
        stride *= iShape[i];
    }

    dnnStatus = cudnnSetTensorNdDescriptor(_tensorDesc, CUDNN_DATA_FLOAT, (int)iShape.size(), iShape.data(), iStride.data());
    if (dnnStatus != CUDNN_STATUS_SUCCESS) {
        free();
        return cudaErrorUnknown;
    }

    _elementSize = elementSize;
    _shape = shape;
    _numElements = numElements;
    _size = size;
    return cudaSuccess;
}

bool Tensor::loadfp32(const char* filename)
{
    if (_host == nullptr)
        return false;

    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Tensor::loadfp32: Error can't open %s\n", filename);
        return false;
    }

    fseek(file, 0, SEEK_END);
    size_t len = ftell(file);
    fseek(file, 0, SEEK_SET);
    if (len != _size)
    {
        printf("Tensor::loadfp32: Error wring file size %s\n", filename);
        fclose(file);
        return false;
    }
    fread(_host, _size, 1, file);
    fclose(file);
    toDevice();
    return true;
}

void Tensor::toDevice()
{
    cudaError err = cudaMemcpy((void*)_device, (const void*)_host, _size, cudaMemcpyKind::cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);
}

void Tensor::toHost()
{
    cudaError err = cudaMemcpy((void*)_host, (const void*)_device, _size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);
}

void Tensor::zeros()
{
    if (_host)
        cudaMemset(_host, 0, _size);
}

void Tensor::fill(const float value)
{
    if (_host) {
        std::fill_n((float*)_host, _numElements, value);
    }
}

void Tensor::fill(const Tensor& value)
{
    if (_host) {
        assert(ndimension() == 2 && value.ndimension() == 1);
        assert(_shape[1] == value.shape()[0]);
        const size_t srcSize = _shape[1] * sizeof(float);
        const float* src = value.host();
        float* dst = host();
        float* dstEnd = dst + _numElements;
        for (; dst < dstEnd; dst += _shape[1])
        {
            memcpy(dst, src, srcSize);
        }
    }
}

void Tensor::free()
{
    if (_host) {
        cudaFreeHost(_host);
        _host = nullptr;
    }
    if (_device) {
        cudaFree(_device);
        _device = nullptr;
    }
    if (_tensorDesc) {
        cudnnDestroyTensorDescriptor(_tensorDesc);
        _tensorDesc = nullptr;
    }
    _shape.clear();
    _numElements = 0;
    _size = 0;
}

void Tensor::print() const
{
    switch (ndimension())
    {
    case 1: {
        const float* val = _host;
        int n = (int)_shape[0];
        for (int i = 0; i < n; ++i) {
            if (i == 0)
                printf("[%f", *val++);
            else
                printf(", %f", *val++);
        }
        printf("]\n");
    } break;

    case 2:
        const float* val = _host;
        int m = (int)_shape[0];
        int n = (int)_shape[1];
        for (int i = 0; i < m; ++i) {
            if (i == 0)
                printf("[");
            for (int j = 0; j < n; ++j) {
                if (j == 0)
                    printf("[%f", *val++);
                else
                    printf(", %f", *val++);
            }
            if (i + 1 == m)
                printf("]]\n");
            else
                printf("],\n");
        }
        break;
    };
}

