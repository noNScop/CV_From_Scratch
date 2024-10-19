#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <memory>

class Module
{
public:
    // Constructor
    Module () : training(false) {}

    // params setter
    void register_parameters(const std::vector<std::shared_ptr<torch::Tensor>>& parameters)
    {
        for (const auto& p : parameters)
        {
            params.push_back(p);
        }
    }

    // children setter
    void register_modules(const std::vector<std::shared_ptr<Module>>& modules)
    {
        for (const auto& m : modules)
        {
            children.push_back(m);
        }
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
        for (std::shared_ptr<Module>& child : children)
        {
            child->set_training(train);
        }
    }

    // Get all parameters recursively
    std::vector<std::shared_ptr<torch::Tensor>> parameters()
    {
        // copy params, to safely add params from children
        std::vector<std::shared_ptr<torch::Tensor>> all_params = params;
        // auto = std::shared_ptr<Module>
        for (auto& child : children)
        {
            auto child_params = child->parameters();
            all_params.insert(all_params.end(), child_params.begin(), child_params.end());
        }
        return all_params;
    }

    // Overloading operator() to mimic Python's __call__
    template <typename... Args> // template parameter pack
    auto operator()(Args&&... args) // perfect forwarding
    {
        // Call forward() and return the result
        return forward(std::forward<Args>(args)...);
    }

protected:
    // virtual function is overridden by derived class
    virtual torch::Tensor forward() = 0;

private:
    std::vector<std::shared_ptr<torch::Tensor>> params;
    std::vector<std::shared_ptr<Module>> children;
    bool training;
};

int main()
{
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
    tensor = torch::matmul(tensor, tensor.t());
    std::cout << tensor << std::endl;
}