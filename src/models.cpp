#include "models.h"

std::shared_ptr<Sequential> MnistCNN::conv_block(int in_channels, int out_channels, int kernel_size, bool activation)
{
    std::shared_ptr<Conv2d> conv =
        std::make_shared<Conv2d>(Conv2d(in_channels, out_channels, kernel_size, 2, (int)kernel_size / 2, !activation));
    std::shared_ptr<BatchNorm2d> batch_norm = std::make_shared<BatchNorm2d>(BatchNorm2d(out_channels));
    std::shared_ptr<Sequential> sequential;

    if (activation)
    {
        std::shared_ptr<ReLU> relu = std::make_shared<ReLU>(ReLU());
        sequential = std::make_shared<Sequential>(Sequential({conv, batch_norm, relu}));
    }
    else
    {
        sequential = std::make_shared<Sequential>(Sequential({conv, batch_norm}));
    }

    register_modules({sequential});
    return sequential;
}

torch::Tensor MnistCNN::forward(torch::Tensor x)
{
    return cnn(x).view({-1, 10});
}