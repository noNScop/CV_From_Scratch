#ifndef LAYERS_H
#define LAYERS_H

#include <memory>
#include <torch/torch.h>
#include <vector>

class Module
{
  public:
    Module();

    // training getter
    bool is_training() const;

    // training setter
    void set_training(bool train);

    // Get all parameters recursively
    std::vector<torch::Tensor> parameters() const;

    // children modules getter
    std::vector<std::shared_ptr<Module>> get_children() const;

    // Overloading operator() to mimic Python's __call__
    template <typename... Args> torch::Tensor operator()(Args &&...args); // perfect forwarding

  protected:
    // pure virtual method is overridden by derived class
    virtual torch::Tensor forward(torch::Tensor x) = 0;

    // params setter
    void register_parameters(const std::initializer_list<torch::Tensor> parameters);

    // children setter
    void register_modules(const std::initializer_list<std::shared_ptr<Module>> modules);

  private:
    std::vector<torch::Tensor> params;
    std::vector<std::shared_ptr<Module>> children;
    bool training;
};

class Linear : public Module
{
  public:
    // ni - number of input features, nf - number of output features
    Linear(int in_channels, int out_channels, bool use_xavier = false, bool use_bias = true);
    
    torch::Tensor weights;
    torch::Tensor bias;

  private:
    bool use_bias;

    torch::Tensor forward(torch::Tensor x) override;
};

// classic convolutional layer
class Conv2d : public Module
{
  public:
    Conv2d(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0, bool use_xavier = false,
           bool use_bias = true);

    torch::Tensor weights;
    torch::Tensor bias;
    
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

    torch::Tensor forward(torch::Tensor x) override;
};

class BatchNorm2d : public Module
{
  public:
    BatchNorm2d(int in_channels, bool running_stats = true, bool zero_init = false, float eps = 1e-5, float momentum = 0.1);

  private:
    torch::Tensor gamma;
    torch::Tensor beta;
    torch::Tensor running_mean;
    torch::Tensor running_var;
    int in_channels;
    float eps;
    float momentum;
    bool running_stats;

    torch::Tensor mean;
    torch::Tensor var;

    torch::Tensor forward(torch::Tensor x) override;
};

class Sequential : public Module
{
  public:
    Sequential(std::initializer_list<std::shared_ptr<Module>> layers);

  private:
    torch::Tensor forward(torch::Tensor x) override;
};

class ReLU : public Module
{
    torch::Tensor forward(torch::Tensor x) override;
};

// Overloading operator() to mimic Python's __call__
template <typename... Args> torch::Tensor Module::operator()(Args &&...args) // perfect forwarding
{
    // Call forward() and return the result
    return forward(std::forward<Args>(args)...);
}

#endif