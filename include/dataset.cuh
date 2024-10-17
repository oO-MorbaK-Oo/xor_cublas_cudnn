#ifndef CUDNN_TEST_DATASET
#define CUDNN_TEST_DATASET
#pragma once

#include "Tensor.cuh"

class Dataset {
public:
    Dataset() {}
    virtual ~Dataset() {}

    size_t len() const { return _x.shape()[0]; };
    const Tensor& x() const { return _x; };
    const Tensor& y() const { return _y; };
protected:
    Tensor _x;
    Tensor _y;
};

#endif // CUDNN_TEST_LAYER