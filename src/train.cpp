#include "train.h"

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

void Learner::train(DataLoader &train_dl, DataLoader &valid_dl, int epochs)
{
    SGD optimizer(model->parameters());
    for (int i = 0; i < epochs; ++i)
    {
        train_loss = 0;
        train_accuracy = 0;
        valid_loss = 0;
        valid_accuracy = 0;
        train_step(train_dl, optimizer);
        valid_step(valid_dl);
        std::cout << "Epoch: " << i + 1 << " train_loss: " << train_loss << " train_acc: " << train_accuracy
                  << " valid_loss: " << valid_loss << " valid_acc: " << valid_accuracy << std::endl;
    }
}

void Learner::train_step(DataLoader &train_dl, SGD &optimizer)
{
    namespace F = torch::nn::functional;

    float batch_accuracy = 0; // batch accuracy accumulator
    float batch_loss = 0;     // batch loss accumulator
    int iters = 0;
    model->set_training(true);

    for (auto &&batch : train_dl)
    {
        ++iters;
        output = model->forward(batch.data);
        loss = F::cross_entropy(output, batch.target);

        loss.backward();
        optimizer.step();
        optimizer.zero_grad();

        batch_loss += loss.item<float>();
        batch_accuracy += (output.argmax(1) == batch.target).sum().template item<float>() / batch.data.size(0);
    }

    train_loss = batch_loss / iters;
    train_accuracy = batch_accuracy / iters;
}

void Learner::valid_step(DataLoader &valid_dl)
{
    namespace F = torch::nn::functional;

    torch::NoGradGuard nograd;
    float batch_accuracy = 0; // batch accuracy accumulator
    float batch_loss = 0;     // batch loss accumulator
    int iters = 0;
    model->set_training(false);

    for (auto &&batch : valid_dl)
    {
        ++iters;
        output = model->forward(batch.data);
        loss = F::cross_entropy(output, batch.target);

        batch_loss += loss.item<float>();
        batch_accuracy += (output.argmax(1) == batch.target).sum().template item<float>() / batch.data.size(0);
    }

    valid_loss = batch_loss / iters;
    valid_accuracy = batch_accuracy / iters;
}