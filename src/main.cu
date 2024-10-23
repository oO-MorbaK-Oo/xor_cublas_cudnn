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

int main(int argc, char** argv)
{
    // read command line arguments
    int size = 100;
    float noise = 0.2f;
    bool verbose = false;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "--size") == 0) || strcmp(argv[i], "-s") == 0) {
            if (i + 1 >= argc)
                break;
            ++i;
            sscanf(argv[i], "%i", &size);
        }
        else if ((strcmp(argv[i], "--noise") == 0) || strcmp(argv[i], "-n") == 0) {
            if (i + 1 >= argc)
                break;
            ++i;
            sscanf(argv[i], "%f", &noise);
        }
        else if ((strcmp(argv[i], "--verbose") == 0) || strcmp(argv[i], "-v") == 0) {
            verbose = true;
        }
    }

    // check arguments validity
    if (size < 1)
        size = 100;
    printf("dataset size: %i\n", size);
    printf("noise amplitude: %f\n", noise);

    // initialize cuda, cublas and cudnn
    Global::initialize();
    {
        // create the dataset
        XORDataset xor_dataset(size, noise);
        // get inputs
        const Tensor& x_test = xor_dataset.x();
        // get outputs
        const Tensor& y_test = xor_dataset.y();

        // load nn weights
        Linear linear1(2, 2, "linear1_weight.fp32", "linear1_bias.fp32");
        Linear linear2(2, 1, "linear2_weight.fp32", "linear2_bias.fp32");
        Activation sigmoid;

        // evaluate the model
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
            if (verbose) {
                float x0 = x_test.host()[2 * i];
                float x1 = x_test.host()[2 * i + 1];
                printf("%.2f %.2f => %i( %.2f)\n", x0, x1, (int)pred, y_hat.host()[i]);
            }
        }

        printf("Accuracy: %f %%\n", float(count * 100) / float(y_test.numel()));
    }
    // free cuda, cublas and cudnn
    Global::shutdown();
    return 0;
}

