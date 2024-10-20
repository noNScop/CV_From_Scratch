#include <iostream>
#include <memory>
#include <torch/torch.h>
#include <vector>

class Module
{
  public:
    Module() : training(true)
    {
    }

    // training getter
    bool is_training() const
    {
        return training;
    }

    // training setter
    void set_training(bool train)
    {
        training = train;
        for (std::shared_ptr<Module> &child : children)
        {
            child->set_training(train);
        }
    }

    // Get all parameters recursively
    std::vector<std::shared_ptr<torch::Tensor>> parameters() const
    {
        // copy params, to safely add params from children
        std::vector<std::shared_ptr<torch::Tensor>> all_params = params;
        // auto = std::shared_ptr<Module>
        for (const auto &child : children)
        {
            auto child_params = child->parameters();
            all_params.insert(all_params.end(), child_params.begin(), child_params.end());
        }
        return all_params;
    }

    // children modules getter
    std::vector<std::shared_ptr<Module>> get_children() const
    {
        return children;
    }

    // Overloading operator() to mimic Python's __call__
    template <typename... Args> auto operator()(Args &&...args) // perfect forwarding
    {
        // Call forward() and return the result
        return forward(std::forward<Args>(args)...);
    }

  protected:
    // pure virtual method is overridden by derived class
    virtual torch::Tensor forward(torch::Tensor x) = 0;

    // params setter
    void register_parameters(const std::initializer_list<std::shared_ptr<torch::Tensor>> parameters)
    {
        params.insert(params.end(), parameters.begin(), parameters.end());
    }

    // children setter
    void register_modules(const std::initializer_list<std::shared_ptr<Module>> modules)
    {
        children.insert(children.end(), modules.begin(), modules.end());
    }

  private:
    std::vector<std::shared_ptr<torch::Tensor>> params;
    std::vector<std::shared_ptr<Module>> children;
    bool training;
};

class Linear : public Module
{
  public:
    // ni - number of input features, nf - number of output features
    Linear(int in_channels, int out_channels, bool use_xavier = false, bool use_bias = true)
    {
        weights =
            std::make_shared<torch::Tensor>(torch::zeros({out_channels, in_channels}, torch::requires_grad(true)));

        // xavier is best for sigmoid, tanh, softmax activations
        // kaiming is best for ReLU
        if (use_xavier)
        {
            torch::nn::init::xavier_normal_(*weights);
        }
        else // use kaiming
        {
            torch::nn::init::kaiming_normal_(*weights);
        }

        if (use_bias)
        {
            bias = std::make_shared<torch::Tensor>(torch::zeros({out_channels}, torch::requires_grad(true)));
            register_parameters({weights, bias});
        }
        else
        {
            bias = nullptr;
            register_parameters({weights});
        }
    }

  private:
    std::shared_ptr<torch::Tensor> weights;
    std::shared_ptr<torch::Tensor> bias;

    torch::Tensor forward(torch::Tensor x) override
    {
        if (bias)
        {
            return torch::matmul(x, weights->t()) + *bias;
        }
        else
        {
            return torch::matmul(x, weights->t());
        }
    }
};

// classic convolutional layer
class Conv2d : public Module
{
  public:
    Conv2d(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0, bool use_xavier = false,
           bool use_bias = true)
        : out_channels(out_channels), kernel_size(kernel_size), stride(stride), padding(padding)
    {
        weights = std::make_shared<torch::Tensor>(
            torch::zeros({out_channels, in_channels, kernel_size, kernel_size}, torch::requires_grad(true)));

        // xavier is best for sigmoid, tanh, softmax activations
        // kaiming is best for ReLU
        if (use_xavier)
        {
            torch::nn::init::xavier_normal_(*weights);
        }
        else // use kaiming
        {
            torch::nn::init::kaiming_normal_(*weights);
        }

        if (use_bias)
        {
            bias = std::make_shared<torch::Tensor>(torch::zeros({out_channels}, torch::requires_grad(true)));
            register_parameters({weights, bias});
        }
        else
        {
            bias = nullptr;
            register_parameters({weights});
        }
    }

  private:
    std::shared_ptr<torch::Tensor> weights;
    std::shared_ptr<torch::Tensor> bias;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;

    int batch_size;
    int height;
    int width;
    int output_height;

    torch::Tensor forward(torch::Tensor x) override
    {
        namespace F = torch::nn::functional;

        batch_size = x.sizes()[0];
        height = x.sizes()[2];
        width = x.sizes()[3];

        // unfold creates tensor that allows applying convolution by matrix multiplication with flattened kernels
        x = F::unfold(x, F::UnfoldFuncOptions({kernel_size, kernel_size}).padding(padding).stride(stride));
        x = torch::matmul(weights->view({out_channels, -1}), x); // flatten the weights

        output_height = (int)((height + 2 * padding - kernel_size) / stride) + 1;
        x = x.view({batch_size, out_channels, output_height, -1});

        if (bias)
        {
            x = x + bias->view({1, out_channels, 1, 1});
        }

        return x;
    }
};

class BatchNorm2d : public Module
{
  public:
    BatchNorm2d(int in_channels, bool zero_init = false, float eps = 1e-5, float momentum = 0.1)
        : in_channels(in_channels), eps(eps), momentum(momentum)
    {
        // initialising gamma with zeros is usefull for residual connections
        if (zero_init)
        {
            gamma = std::make_shared<torch::Tensor>(torch::zeros({in_channels}, torch::requires_grad(true)));
        }
        else
        {
            gamma = std::make_shared<torch::Tensor>(torch::ones({in_channels}, torch::requires_grad(true)));
        }

        beta = std::make_shared<torch::Tensor>(torch::zeros({in_channels}, torch::requires_grad(true)));
        running_mean = std::make_shared<torch::Tensor>(torch::zeros({in_channels}));
        running_var = std::make_shared<torch::Tensor>(torch::ones({in_channels}));
        register_parameters({gamma, beta, running_mean, running_var});
    }

  private:
    std::shared_ptr<torch::Tensor> gamma;
    std::shared_ptr<torch::Tensor> beta;
    std::shared_ptr<torch::Tensor> running_mean;
    std::shared_ptr<torch::Tensor> running_var;
    int in_channels;
    float eps;
    float momentum;

    torch::Tensor mean;
    torch::Tensor var;

    torch::Tensor forward(torch::Tensor x) override
    {
        if (is_training())
        {
            // calculate statistics for each channel across batch and spatial dimensions
            mean = x.mean({0, 2, 3}, true);       // keepdim = true
            var = x.var({0, 2, 3}, false, true);  // unbiased = false, keepdim = true
            x = (x - mean) / (var + eps).sqrt_(); // in place sqrt for better performance

            var = x.var({0, 2, 3}, true); // unbiased = true, keepdim = false (ubiased var is needed for running stats)
            *running_mean = (1 - momentum) * *running_mean + momentum * mean.view({in_channels});
            *running_var = (1 - momentum) * *running_var + momentum * var;
        }
        else
        {
            // in place sqrt would be stored in running_var, we don't want that
            x = (x - running_mean->view({1, in_channels, 1, 1})) /
                (running_var->view({1, in_channels, 1, 1}) + eps).sqrt();
        }

        return gamma->view({1, in_channels, 1, 1}) * x + beta->view({1, in_channels, 1, 1});
    }
};

class Sequential : public Module
{
  public:
    Sequential(std::initializer_list<std::shared_ptr<Module>> layers)
    {
        register_modules(layers);
    }

  private:
    torch::Tensor forward(torch::Tensor x) override
    {
        // std::vector<std::shared_ptr<Module>> children;
        for (const std::shared_ptr<Module> &module : get_children())
        {
            x = (*module)(x);
        }

        return x;
    }
};