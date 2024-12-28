#ifndef MODELS_H
#define MODELS_H

#include "layers.h"
#include "tensor.h"
#include <cereal/archives/binary.hpp>
#include <memory>

// MnistCNN expects inputs of (batch_size, 1, 28, 28), as mnist images have a single color channel
class MnistCNN : public Module
{
  public:
    Tensor<float> forward(Tensor<float> x) override;

    template <class Archive> void serialize(Archive &ar)
    {
        ar(cereal::base_class<Module>(this), cnn);
    }

  private:
    std::shared_ptr<Sequential> conv_block(int in_channels, int out_channels, int kernel_size = 3,
                                           bool activation = true);

    Sequential cnn = Sequential({
        conv_block(1, 8, 5),          // 14 x 14
        conv_block(8, 16),            // 7 x 7
        conv_block(16, 32),           // 4 x 4
        conv_block(32, 64),           // 2 x 2
        conv_block(64, 10, 3, false), // 1 x 1
    });
};

#endif