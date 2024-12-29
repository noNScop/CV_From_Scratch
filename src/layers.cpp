#include "layers.h"

Module::Module() : training(true)
{
}

bool Module::is_training() const
{
    return training;
}

void Module::set_training(bool train)
{
    training = train;
    for (std::shared_ptr<Module> &child : children)
    {
        child->set_training(train);
    }
}

std::vector<Tensor<float>> Module::parameters() const
{
    std::vector<Tensor<float>> all_params = params;
    for (const auto &child : children)
    {
        auto child_params = child->parameters();
        all_params.insert(all_params.end(), child_params.begin(), child_params.end());
    }
    return all_params;
}

std::vector<std::shared_ptr<Module>> Module::get_children() const
{
    return children;
}

void Module::register_parameters(const std::initializer_list<Tensor<float>> parameters)
{
    params.insert(params.end(), parameters.begin(), parameters.end());
}

void Module::register_modules(const std::initializer_list<std::shared_ptr<Module>> modules)
{
    children.insert(children.end(), modules.begin(), modules.end());
}

Linear::Linear(int in_channels, int out_channels, bool use_xavier, bool use_bias)
    : use_bias(use_bias)
{
    std::vector<size_t> shape = {static_cast<size_t>(out_channels), static_cast<size_t>(in_channels)};

    if (use_xavier)
    {
        weights = Tensor<float>::xavier_normal(shape, 1, true);
    }
    else 
    {
        weights = Tensor<float>::kaiming_normal(shape, true);
    }

    if (use_bias)
    {
        bias = Tensor<float>::zeros({static_cast<size_t>(out_channels)}, true);
        register_parameters({weights, bias});
    }
    else
    {
        register_parameters({weights});
    }
}

Tensor<float> Linear::forward(Tensor<float> x)
{
    if (use_bias)
    {
        Tensor<float> transpose = weights.transpose(0, 1);
        Tensor<float> mul = Tensor<float>::matmul(x, transpose);
        return mul + bias;
    }
    else
    {
        Tensor<float> transpose = weights.transpose(0, 1);
        return Tensor<float>::matmul(x, transpose);
    }
}

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding, bool use_xavier,
               bool use_bias)
    : out_channels(out_channels), kernel_size(kernel_size), stride(stride), padding(padding), use_bias(use_bias)
{
    std::vector<size_t> shape = {
        static_cast<size_t>(out_channels), 
        static_cast<size_t>(in_channels), 
        static_cast<size_t>(kernel_size), 
        static_cast<size_t>(kernel_size)
    };

    if (use_xavier)
    {
        weights = Tensor<float>::xavier_normal(shape, 1, true);
    }
    else 
    {
        weights = Tensor<float>::kaiming_normal(shape, true);
    }

    if (use_bias)
    {
        bias = Tensor<float>::zeros({static_cast<size_t>(out_channels)}, true);
        register_parameters({weights, bias});
    }
    else
    {
        register_parameters({weights});
    }
}

Tensor<float> Conv2d::forward(Tensor<float> x)
{
    batch_size = x.size()[0];
    height = x.size()[2];
    width = x.size()[3];

    x = Tensor<float>::unfold(x, kernel_size, padding, stride);
    Tensor<float> weights_view = weights.view({out_channels, -1});
    x = Tensor<float>::matmul(weights_view, x); // flatten the weights

    output_height = (int)((height + 2 * padding - kernel_size) / stride) + 1;
    x = x.view({batch_size, out_channels, output_height, -1});

    if (use_bias)
    {
        Tensor<float> bias_view = bias.view({1, out_channels, 1, 1});
        x = x + bias_view;
    }

    return x;
}

BatchNorm2d::BatchNorm2d(int in_channels, bool zero_init, float eps, float momentum)
    : in_channels(in_channels), eps(eps), momentum(momentum)
{
    // Initialising gamma with zeros is usefull for residual connections
    if (zero_init)
    {
        gamma = Tensor<float>::zeros({static_cast<size_t>(in_channels)}, true);
    }
    else
    {
        gamma = Tensor<float>::ones({static_cast<size_t>(in_channels)}, true);
    }

    beta = Tensor<float>::zeros({static_cast<size_t>(in_channels)}, true);
    running_mean = Tensor<float>::zeros({static_cast<size_t>(in_channels)});
    running_var = Tensor<float>::zeros({static_cast<size_t>(in_channels)});
    register_parameters({gamma, beta});
}

Tensor<float> BatchNorm2d::forward(Tensor<float> x)
{
    if (is_training())
    {
        mean = x.mean({0, 2, 3}, true);
        var = x.var({0, 2, 3}, false);
        Tensor<float> var_view = var.view({1, -1, 1, 1});
        Tensor<float> var_plus_eps = var_view + eps;
        Tensor<float> sqrt_var_plus_eps = var_plus_eps.sqrt();
        Tensor<float> x_minus_mean = x - mean;
        x = x_minus_mean / sqrt_var_plus_eps;

        Tensor<float> mean_view = mean.view({in_channels});
        Tensor<float> momentum_times_mean = mean_view * momentum;
        Tensor<float> temp = running_mean * (1 - momentum);
        running_mean = temp + momentum_times_mean;

        Tensor<float> momentum_times_var = var * momentum;
        Tensor<float> temp2 = running_var * (1 - momentum);
        running_var = temp2 + momentum_times_var;
    }
    else
    {
        Tensor<float> r_var_view = running_var.view({1, in_channels, 1, 1});
        Tensor<float> r_var_plus_eps = r_var_view + eps;
        Tensor<float> sqrt_r_var_plus_eps = r_var_plus_eps.sqrt();
        Tensor<float> r_mean_view = running_mean.view({1, in_channels, 1, 1});
        Tensor<float> x_minus_r_mean = x - r_mean_view;
        x = x_minus_r_mean / sqrt_r_var_plus_eps;
    }
    Tensor<float> gamma_view = gamma.view({1, in_channels, 1, 1});
    Tensor<float> beta_view = beta.view({1, in_channels, 1, 1});
    Tensor<float> mul = gamma_view * x;
    return mul + beta_view;
}

Sequential::Sequential(std::initializer_list<std::shared_ptr<Module>> layers)
{
    register_modules(layers);
}

Tensor<float> Sequential::forward(Tensor<float> x)
{
    for (const std::shared_ptr<Module> &module : get_children())
    {
        x = module->forward(x);
    }

    return x;
}

Tensor<float> ReLU::forward(Tensor<float> x)
{
    return Tensor<float>::relu(x);
}