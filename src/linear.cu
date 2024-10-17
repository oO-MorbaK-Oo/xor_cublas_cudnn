#include "Linear.cuh"

Linear::Linear(const size_t inFeatures, const size_t outFeatures, const bool bias)
    : _inFeatures(inFeatures), _outFeatures(outFeatures), _bBias(bias)
{
    _weight.alloc({ _outFeatures, _inFeatures });
    if (_bBias) {
        _bias.alloc({ _outFeatures });
    }

    // fill tensor with random values
    float* data = _weight.host();
    thrust::default_random_engine rng(0);
    thrust::random::normal_distribution<float> dist(-1.0f, 1.0f);
    for (int n = 0; n < _weight.numel(); ++n) {
        data[n] = dist(rng);
    }

    data = _bias.host();
    for (int n = 0; n < _bias.numel(); ++n) {
        data[n] = dist(rng);
    }

    _weight.toDevice();
    _bias.toDevice();
}

Linear::Linear(const size_t inFeatures, const size_t outFeatures, const float* weights, const float* bias)
    : _inFeatures(inFeatures), _outFeatures(outFeatures), _bBias(bias != nullptr)
{
    _weight.alloc({ _outFeatures, _inFeatures });
    if (_bBias) {
        _bias.alloc({ _outFeatures });
    }

    if (weights)
        memcpy(_weight.host(), weights, _weight.numel() * sizeof(float));
    if (bias)
        memcpy(_bias.host(), bias, _bias.numel() * sizeof(float));
    _weight.toDevice();
    _bias.toDevice();
}

Linear::Linear(const size_t inFeatures, const size_t outFeatures, const char* weightsFile, const char* biasFile)
    : _inFeatures(inFeatures), _outFeatures(outFeatures), _bBias(biasFile != nullptr)
{
    _weight.alloc({ _outFeatures, _inFeatures });
    if (_bBias) {
        _bias.alloc({ _outFeatures });
    }

    if (weightsFile)
        _weight.loadfp32(weightsFile);

    if (biasFile)
        _bias.loadfp32(biasFile);

    _weight.toDevice();
    _bias.toDevice();

}



Linear::~Linear()
{
}


Tensor& Linear::forward(const Tensor& x, Tensor& y)
{
    float alpha = 1.0f;
    float beta = 1.0f;

    y.alloc({ x.shape()[0],_outFeatures });
    y.fill(_bias);
    y.toDevice();

    size_t m = _weight.shape()[0];
    size_t n = x.shape()[1];
    size_t k = x.shape()[0];
    size_t lda = _weight.shape()[1];
    size_t ldb = x.shape()[1];
    size_t ldc = y.shape()[1];
    auto ret = cublasSgemm(Global::getCublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N,
        (int)m, (int)k, (int)n,
        &alpha,
        _weight.device(), (int)lda,
        x.device(), (int)ldb,
        &beta,
        y.device(), (int)ldc);
    return y;
}

