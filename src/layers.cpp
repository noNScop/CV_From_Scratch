#include "layers.h"
#include <memory>
#include <torch/torch.h>
#include <vector>

Module::Module() : training(true)
{
}

// training getter
bool Module::is_training() const
{
    return training;
}

// training setter
void Module::set_training(bool train)
{
    training = train;
    for (std::shared_ptr<Module> &child : children)
    {
        child->set_training(train);
    }
}

// Get all parameters recursively
std::vector<torch::Tensor> Module::parameters() const
{
    // copy params, to safely add params from children
    std::vector<torch::Tensor> all_params = params;
    // auto = std::shared_ptr<Module>
    for (const auto &child : children)
    {
        auto child_params = child->parameters();
        all_params.insert(all_params.end(), child_params.begin(), child_params.end());
    }
    return all_params;
}

// children modules getter
std::vector<std::shared_ptr<Module>> Module::get_children() const
{
    return children;
}

// params setter
void Module::register_parameters(const std::initializer_list<torch::Tensor> parameters)
{
    params.insert(params.end(), parameters.begin(), parameters.end());
}

// children setter
void Module::register_modules(const std::initializer_list<std::shared_ptr<Module>> modules)
{
    children.insert(children.end(), modules.begin(), modules.end());
}

// ni - number of input features, nf - number of output features
Linear::Linear(int in_channels, int out_channels, bool use_xavier, bool use_bias) : use_bias(use_bias)
{
    weights = torch::zeros({out_channels, in_channels}, torch::requires_grad(true));

    // xavier is best for sigmoid, tanh, softmax activations
    // kaiming is best for ReLU
    if (use_xavier)
    {
        torch::nn::init::xavier_normal_(weights);
    }
    else // use kaiming
    {
        torch::nn::init::kaiming_normal_(weights);
    }

    if (use_bias)
    {
        bias = torch::zeros({out_channels}, torch::requires_grad(true));
        register_parameters({weights, bias});
    }
    else
    {
        register_parameters({weights});
    }
}

torch::Tensor Linear::forward(torch::Tensor x)
{
    if (use_bias)
    {
        return torch::matmul(x, weights.t()) + bias;
    }
    else
    {
        return torch::matmul(x, weights.t());
    }
}

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding, bool use_xavier,
               bool use_bias)
    : out_channels(out_channels), kernel_size(kernel_size), stride(stride), padding(padding), use_bias(use_bias)
{
    weights = torch::zeros({out_channels, in_channels, kernel_size, kernel_size}, torch::requires_grad(true));

    // xavier is best for sigmoid, tanh, softmax activations
    // kaiming is best for ReLU
    if (use_xavier)
    {
        torch::nn::init::xavier_normal_(weights);
    }
    else // use kaiming
    {
        torch::nn::init::kaiming_normal_(weights);
    }

    if (use_bias)
    {
        bias = torch::zeros({out_channels}, torch::requires_grad(true));
        register_parameters({weights, bias});
    }
    else
    {
        register_parameters({weights});
    }
}

torch::Tensor Conv2d::forward(torch::Tensor x)
{
    namespace F = torch::nn::functional;

    batch_size = x.sizes()[0];
    height = x.sizes()[2];
    width = x.sizes()[3];

    // unfold creates tensor that allows applying convolution by matrix multiplication with flattened kernels
    x = F::unfold(x, F::UnfoldFuncOptions({kernel_size, kernel_size}).padding(padding).stride(stride));
    x = torch::matmul(weights.view({out_channels, -1}), x); // flatten the weights

    output_height = (int)((height + 2 * padding - kernel_size) / stride) + 1;
    x = x.view({batch_size, out_channels, output_height, -1});

    if (use_bias)
    {
        x = x + bias.view({1, out_channels, 1, 1});
    }

    return x;
}

BatchNorm2d::BatchNorm2d(int in_channels, bool running_stats, bool zero_init, float eps, float momentum)
    : in_channels(in_channels), eps(eps), momentum(momentum), running_stats(running_stats)
{
    // initialising gamma with zeros is usefull for residual connections
    if (zero_init)
    {
        gamma = torch::zeros({in_channels}, torch::requires_grad(true));
    }
    else
    {
        gamma = torch::ones({in_channels}, torch::requires_grad(true));
    }

    beta = torch::zeros({in_channels}, torch::requires_grad(true));
    running_mean = torch::zeros({in_channels});
    running_var = torch::zeros({in_channels});
    register_parameters({gamma, beta});
}

torch::Tensor BatchNorm2d::forward(torch::Tensor x)
{
    if (is_training() || !running_stats)
    {
        // calculate statistics for each channel across batch and spatial dimensions
        mean = x.mean({0, 2, 3}, true);      // keepdim = true
        var = x.var({0, 2, 3}, false, true); // unbiased = false, keepdim = true
        x = (x - mean) / (var + eps).sqrt();

        var = x.var({0, 2, 3}, true); // unbiased = true, keepdim = false (ubiased var is needed for running stats)
        running_mean = (1 - momentum) * running_mean + momentum * mean.view({in_channels});
        running_var = (1 - momentum) * running_var + momentum * var;
    }
    else
    {
        x = (x - running_mean.view({1, in_channels, 1, 1})) / (running_var.view({1, in_channels, 1, 1}) + eps).sqrt();
    }

    return gamma.view({1, in_channels, 1, 1}) * x + beta.view({1, in_channels, 1, 1});
}

Sequential::Sequential(std::initializer_list<std::shared_ptr<Module>> layers)
{
    register_modules(layers);
}

torch::Tensor Sequential::forward(torch::Tensor x)
{
    // std::vector<std::shared_ptr<Module>> children;
    for (const std::shared_ptr<Module> &module : get_children())
    {
        x = (*module)(x);
    }

    return x;
}

torch::Tensor ReLU::forward(torch::Tensor x)
{
    namespace F = torch::nn::functional;
    return F::relu(x);
}