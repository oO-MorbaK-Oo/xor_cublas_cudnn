#include <stdio.h>
#include "Linear.cuh"
#include "Activation.cuh"
#include "Dataset.cuh"

#pragma comment(lib, "cudnn.lib")
#pragma comment(lib, "cublas.lib")


// create a XOR dataset with some random noise
class XORDataset : public Dataset
{
public:
    XORDataset(const int numSamples, const float noise)
    {
        _x.alloc({ size_t(numSamples), 2 });
        _y.alloc({ size_t(numSamples), 1 });
        _x.zeros();
        _y.zeros();

        float* x = (float*)_x.host();
        float* y = (float*)_y.host();
        for (int i = 0; i < numSamples / 4; ++i) {
            int i00 = i;
            int i01 = i + numSamples / 4;
            int i10 = i + numSamples / 2;
            int i11 = i + 3 * (numSamples / 4);

            x[2 * i00] = 0.0f;
            x[2 * i00 + 1] = 0.0f;
            y[i00] = 0.0f;

            x[2 * i01] = 0.0f;
            x[2 * i01 + 1] = 1.0f;
            y[i01] = 1.0f;

            x[2 * i10] = 1.0f;
            x[2 * i10 + 1] = 0.0f;
            y[i10] = 1.0f;

            x[2 * i11] = 1.0f;
            x[2 * i11 + 1] = 1.0f;
            y[i11] = 0.0f;
        }

        for (int i = 0; i < numSamples; ++i) {
            x[i] += noise * float(rand()) / float(RAND_MAX);
        }
        _x.toDevice();
        _y.toDevice();
    }

    virtual ~XORDataset()
    {
    }
};

int main()
{
    Global::initialize();
    {
        XORDataset xor_set(100, 0.2f);
        const Tensor& x_test = xor_set.x();
        const Tensor& y_test = xor_set.y();

        // load weights
        Linear linear1(2, 2, "linear1_weight.fp32", "linear1_bias.fp32");
        Linear linear2(2, 1, "linear2_weight.fp32", "linear2_bias.fp32");
        Activation sigmoid;

        Tensor a, b, c, y_hat;
        sigmoid.forward(linear1.forward(x_test, a), b);
        sigmoid.forward(linear2.forward(b, c), y_hat);

        // accuracy
        int count = 0;
        y_hat.toHost();
        for (int i = 0; i < y_test.numel(); ++i) {
            float truth = y_test.host()[i];
            float pred = (y_hat.host()[i] > 0.5) ? 1.0f : 0.0f;
            if (truth == pred)
                ++count;
        }

        printf("Accuracy: %f %%\n", float(count * 100) / float(y_test.numel()));
    }
    Global::shutdown();
    return 0;
}

