#include <iostream>
#include <memory>
#include <torch/torch.h>
#include <vector>

class Module
{
  public:
    // Constructor
    Module() : training(false)
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
        for (auto &child : children)
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
    void register_parameters(std::vector<std::shared_ptr<torch::Tensor>> &&parameters)
    {
        for (const auto &p : parameters)
        {
            params.push_back(p);
        }
    }

    // children setter
    void register_modules(const std::vector<std::shared_ptr<Module>> &modules)
    {
        for (const auto &m : modules)
        {
            children.push_back(m);
        }
    }

  private:
    std::vector<std::shared_ptr<torch::Tensor>> params;
    std::vector<std::shared_ptr<Module>> children;
    bool training;
};

class Linear : public Module
{
  public:
    Linear(int ni, int nf, bool use_xavier = false, bool use_bias = true) : ni(ni), nf(nf)
    {
        weights = std::make_shared<torch::Tensor>(torch::zeros({nf, ni}, torch::requires_grad(true)));

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
            bias = std::make_shared<torch::Tensor>(torch::zeros(nf, torch::requires_grad(true)));
            register_parameters(std::vector<std::shared_ptr<torch::Tensor>>{weights, bias});
        }
        else
        {
            bias = nullptr;
            register_parameters(std::vector<std::shared_ptr<torch::Tensor>>{weights});
        }
    }

  private:
    int ni; // number of input features
    int nf; // number of output features
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

int main()
{
    // Set the seed for reproducibility
    torch::manual_seed(0);

    // Create a linear layer with input size 10 and output size 5
    Linear linearLayer1(10, 5, true, true);
    Linear linearLayer2(10, 10, true, false);
    Linear linearLayer3(10, 10, false, true);
    Linear linearLayer4(10, 10, false, false);

    // Create a random input tensor with requires_grad set to true
    torch::Tensor input = torch::randn({3, 10}, torch::requires_grad(true));

    // Perform a forward pass
    torch::Tensor output = linearLayer1(linearLayer2(linearLayer3(linearLayer4(input))));

    // Print the output
    std::cout << "Output:\n" << output << std::endl;

    // Define a simple loss function (mean squared error)
    torch::Tensor target = torch::randn({3, 5}); // Random target tensor
    torch::Tensor loss = torch::mean(torch::pow(output - target, 2));

    // Print the loss
    std::cout << "Loss:\n" << loss.item<double>() << std::endl;

    // Backward pass to compute gradients
    loss.backward();

    // Access and print gradients for weights and bias
    std::cout << "Weight gradients:\n" << linearLayer1.parameters()[0]->grad() << std::endl;
    if (linearLayer1.parameters().size() > 1)
    {
        std::cout << "Bias gradients:\n" << linearLayer1.parameters()[1]->grad() << std::endl;
    }

    return 0;
}