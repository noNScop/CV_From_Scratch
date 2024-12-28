#ifndef LAYERS_H
#define LAYERS_H

#include "tensor.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <map>
#include <memory>
#include <vector>

class Module
{
  public:
    Module();

    // Returns state dictionary with parameters as  a map, right now in an initial state,
    // it is supposed to be used for saving tensors, although it may not have all functionality
    // required just yet
    std::map<std::string, Tensor<float>> state_dict() const;

    // training getter
    bool is_training() const;

    // training setter
    void set_training(bool train);

    // Get all parameters recursively
    std::vector<Tensor<float>> parameters() const;

    // children modules getter
    std::vector<std::shared_ptr<Module>> get_children() const;

    // Overloading operator() to mimic Python's __call__
    template <typename... Args> Tensor<float> operator()(Args &&...args); // perfect forwarding

    virtual Tensor<float> forward(Tensor<float> x) = 0;

    template <class Archive> void serialize(Archive &ar)
    {
        ar(training, children);
    }

  protected:
    // params setter
    void register_parameters(const std::initializer_list<Tensor<float>> parameters);

    // children setter
    void register_modules(const std::initializer_list<std::shared_ptr<Module>> modules);

  private:
    std::vector<Tensor<float>> params;
    std::vector<std::shared_ptr<Module>> children;
    bool training;
};

class Linear : public Module
{
  public:
    // ni - number of input features, nf - number of output features
    Linear(int in_channels, int out_channels, bool use_xavier = false, bool use_bias = true);

    Linear() {};

    Tensor<float> weights;
    Tensor<float> bias;

    Tensor<float> forward(Tensor<float> x) override;

    template <class Archive> void serialize(Archive &ar)
    {
        ar(cereal::base_class<Module>(this), weights, bias, use_bias);
        if (!Archive::is_saving::value)
        {
            if (use_bias)
            {
                register_parameters({weights, bias});
            }
            else
            {
                register_parameters({weights});
            }
        }
    }

  private:
    bool use_bias;
};

// classic convolutional layer
class Conv2d : public Module
{
  public:
    Conv2d(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0, bool use_xavier = false,
           bool use_bias = true);

    Conv2d() {};

    Tensor<float> weights;
    Tensor<float> bias;

    Tensor<float> forward(Tensor<float> x) override;

    template <class Archive> void serialize(Archive &ar)
    {
        ar(cereal::base_class<Module>(this), weights, bias, use_bias, out_channels, kernel_size, stride, padding,
           batch_size, height, width, output_height);
        if (!Archive::is_saving::value)
        {
            if (use_bias)
            {
                register_parameters({weights, bias});
            }
            else
            {
                register_parameters({weights});
            }
        }
    }

  private:
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    bool use_bias;

    int batch_size;
    int height;
    int width;
    int output_height;
};

// Batch normalization layer
class BatchNorm2d : public Module
{
  public:
    BatchNorm2d(int in_channels, bool zero_init = false, float eps = 1e-5, float momentum = 0.1);
    BatchNorm2d() {};

    Tensor<float> forward(Tensor<float> x) override;

    template <class Archive> void serialize(Archive &ar)
    {
        ar(cereal::base_class<Module>(this), gamma, beta, running_mean, running_var, in_channels, eps, momentum, mean,
           var);
        if (!Archive::is_saving::value)
        {
            register_parameters({gamma, beta});
        }
    }
  private:
    Tensor<float> gamma;
    Tensor<float> beta;
    Tensor<float> running_mean;
    Tensor<float> running_var;
    int in_channels;
    float eps;
    float momentum;

    Tensor<float> mean;
    Tensor<float> var;
};

class Sequential : public Module
{
  public:
    Sequential(std::initializer_list<std::shared_ptr<Module>> layers);
    Sequential() {};

    Tensor<float> forward(Tensor<float> x) override;

    template <class Archive> void serialize(Archive &ar)
    {
        ar(cereal::base_class<Module>(this));
    }
};

class ReLU : public Module
{
  public:
    Tensor<float> forward(Tensor<float> x) override;
    ReLU() {};

    template <class Archive> void serialize(Archive &ar)
    {
        ar(cereal::base_class<Module>(this));
    }
};

// Overloading operator() to mimic Python's __call__
template <typename... Args> Tensor<float> Module::operator()(Args &&...args) // perfect forwarding
{
    // Call forward() and return the result
    return forward(std::forward<Args>(args)...);
}

CEREAL_REGISTER_TYPE(Sequential);
CEREAL_REGISTER_TYPE(ReLU);
CEREAL_REGISTER_TYPE(Linear);
CEREAL_REGISTER_TYPE(Conv2d)  ;
CEREAL_REGISTER_TYPE(BatchNorm2d);

#endif