#include "train.h"
#include "layers.h"
#include "models.h"
#include <iostream>
#include <torch/torch.h>

Optimizer::Optimizer(std::vector<torch::Tensor> parameters, float learning_rate)
    : parameters(parameters), learning_rate(learning_rate)
{
}

void Optimizer::zero_grad()
{
    for (torch::Tensor &param : parameters)
    {
        if (param.grad().defined())
        {
            param.grad().zero_();
        }
    }
}

SGD::SGD(std::vector<torch::Tensor> parameters, float learning_rate, float momentum)
    : Optimizer(parameters, learning_rate), momentum(momentum)
{
    for (torch::Tensor &param : parameters)
    {
        ema.push_back(torch::zeros_like(param));
    }
}

void SGD::step()
{
    for (int i = 0; i < parameters.size(); ++i)
    {
        if (parameters[i].grad().defined())
        {
            ema[i] = momentum * ema[i] + parameters[i].grad();
            parameters[i].data() -= learning_rate * ema[i];
        }
    }
}

Learner::Learner(std::shared_ptr<Module> model) : model(model)
{
}