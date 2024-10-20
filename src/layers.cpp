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

    // Overloading operator() to mimic Python's __call__
    template <typename... Args>     // template parameter pack
    auto operator()(Args &&...args) // perfect forwarding
    {
        // Call forward() and return the result
        return forward(std::forward<Args>(args)...);
    }

  protected:
    // pure virtual method is overridden by derived class
    virtual torch::Tensor forward(torch::Tensor x) = 0;

    // params setter
    void register_parameters(const std::initializer_list<std::shared_ptr<torch::Tensor>> &&parameters)
    {
        params.insert(params.end(), parameters.begin(), parameters.end());
    }

    // children setter
    void register_modules(const std::initializer_list<std::shared_ptr<Module>> &modules)
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
            x = (x - running_mean->view({1, in_channels, 1, 1})) / (running_var->view({1, in_channels, 1, 1}) + eps).sqrt();
        }

        return gamma->view({1, in_channels, 1, 1}) * x + beta->view({1, in_channels, 1, 1});
    }
};





void test_linear_layer() {
    // Create a Linear layer with input features = 4, output features = 2
    Linear linear_layer(4, 2);

    // Create a sample input tensor of size (batch_size, input_features)
    torch::Tensor input = torch::randn({3, 4});  // 3 batches, 4 input features
    std::cout << "Input Tensor (Linear):\n" << input << std::endl;

    // Pass the input through the Linear layer
    torch::Tensor output = linear_layer(input);

    // Print the output
    std::cout << "Output Tensor (Linear):\n" << output << std::endl;

    // Check the output shape (batch_size, output_features)
    assert(output.sizes() == torch::IntArrayRef({3, 2}));
    std::cout << "Linear layer test passed!" << std::endl;
}

void test_conv_layer() {
    // Create a Conv2d layer with input channels = 3, output channels = 8, kernel size = 3, stride = 1, padding = 1
    Conv2d conv_layer(3, 8, 3, 1, 1);

    // Create a sample input tensor of size (batch_size, channels, height, width)
    torch::Tensor input = torch::randn({1, 3, 32, 32});  // 1 batch, 3 channels, 32x32 image
    std::cout << "Input Tensor (Conv2d):\n" << input.sizes() << std::endl;

    // Pass the input through the Conv2d layer
    torch::Tensor output = conv_layer(input);

    // Print the output
    std::cout << "Output Tensor (Conv2d):\n" << output.sizes() << std::endl;

    // Expected output shape is (batch_size, output_channels, height, width)
    // After convolution with stride 1 and padding 1, the height and width remain unchanged (32x32)
    assert(output.sizes() == torch::IntArrayRef({1, 8, 32, 32}));
    std::cout << "Conv2d layer test passed!" << std::endl;
}

int main()
{
    torch::Tensor tensor = torch::randn({4, 3, 1, 1});
    BatchNorm2d bn(3);
    std::cout << bn.is_training() << std::endl;
    torch::Tensor a = bn(tensor);
    std::cout << tensor - a << std::endl;
    torch::Tensor rmean = *bn.parameters()[2];
    torch::Tensor rvar = *bn.parameters()[3];
    bn.set_training(false);
    a = bn(tensor);
    tensor = (tensor - rmean.view({1, 3, 1, 1})) / (rvar.view({1, 3, 1, 1}) + 1e-5).sqrt();
    std::cout << tensor << std::endl;
    std::cout << a << std::endl;
    std::cout << bn.is_training() << std::endl;

    // Test Linear Layer
    test_linear_layer();

    // Test Conv2d Layer
    test_conv_layer();

    return 0;
}