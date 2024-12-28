#ifndef LAYERS_H
#define LAYERS_H

#include <map>
#include <memory>
#include "tensor.h"
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

    Tensor<float> weights;
    Tensor<float> bias;

    Tensor<float> forward(Tensor<float> x) override;

  private:
    bool use_bias;
};

// classic convolutional layer
class Conv2d : public Module
{
  public:
    Conv2d(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0, bool use_xavier = false,
           bool use_bias = true);

    Tensor<float> weights;
    Tensor<float> bias;

    Tensor<float> forward(Tensor<float> x) override;

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

    Tensor<float> forward(Tensor<float> x) override;

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

    Tensor<float> forward(Tensor<float> x) override;
};

class ReLU : public Module
{
  public:
    Tensor<float> forward(Tensor<float> x) override;
};

// Overloading operator() to mimic Python's __call__
template <typename... Args> Tensor<float> Module::operator()(Args &&...args) // perfect forwarding
{
    // Call forward() and return the result
    return forward(std::forward<Args>(args)...);
}

#endif