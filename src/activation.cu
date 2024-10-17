#include "activation.cuh"

Activation::Activation(cudnnActivationMode_t mode, double coef, cudnnNanPropagation_t reluNanOpt)
{
    cudnnCreateActivationDescriptor(&_desc);
    cudnnSetActivationDescriptor(_desc, mode, reluNanOpt, coef);
}

Activation::~Activation()
{
    cudnnDestroyActivationDescriptor(_desc);
}

Tensor& Activation::forward(const Tensor& x, Tensor& y)
{
    float alpha = 1.0f;
    float beta = 0.0f;

    y.alloc(x.shape());

    cudnnStatus_t status = cudnnActivationForward(Global::getCudnnHandle(),
        _desc,
        &alpha,
        x.desc(),
        x.device(),
        &beta,
        y.desc(),
        y.device());
    return y;
}

