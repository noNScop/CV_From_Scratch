#include "train.h"

Optimizer::Optimizer(std::vector<Tensor<float>> parameters, float learning_rate)
    : parameters(parameters), learning_rate(learning_rate)
{
}

void Optimizer::zero_grad()
{
    for (Tensor<float> &param : parameters)
    {
        param.zero_grad();
    }
}

SGD::SGD(std::vector<Tensor<float>> parameters, float learning_rate, float momentum)
    : Optimizer(parameters, learning_rate), momentum(momentum)
{
    for (Tensor<float> &param : parameters)
    {
        ema.push_back(Tensor<float>::zeros(param.size()));
    }
}

void SGD::step()
{
    NoGradGuard no_grad;
    for (int i = 0; i < parameters.size(); ++i)
    {
        if (parameters[i].grad != nullptr)
        {
            ema[i] = (ema[i] * momentum) + *(parameters[i].grad);
            Tensor<float> add = ema[i] * learning_rate;
            Tensor<float> minus_add = -add;
            parameters[i] += minus_add;
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
    float batch_accuracy = 0; 
    float batch_loss = 0;   
    int iters = 0;
    model->set_training(true);

    for (auto &&batch : train_dl)
    {
        ++iters;
        output = model->forward(batch.data);
        Tensor<int> targets = batch.target.view({-1});
        loss = Tensor<float>::cross_entropy(output, targets);

        loss.backward();
        optimizer.step();
        optimizer.zero_grad();

        NoGradGuard no_grad;
        batch_loss += loss[{0}];
        int correct = 0;
        for (int i = 0; i < batch.data.size()[0]; i++)
        {
            Tensor<float> curr_batch = output[{{i, i + 1}}];
            Tensor<int> argmax = curr_batch.argmax();
            if (argmax[{0}] == targets[{i}])
            {
                correct++;
            }
        }
        batch_accuracy += (float)correct / batch.data.size()[0];
        no_grad.~NoGradGuard();
    }

    train_loss = batch_loss / iters;
    train_accuracy = batch_accuracy / iters;
}

void Learner::valid_step(DataLoader &valid_dl)
{
    NoGradGuard no_grad;
    float batch_accuracy = 0; 
    float batch_loss = 0;   
    int iters = 0;
    model->set_training(false);

    for (auto &&batch : valid_dl)
    {
        ++iters;
        output = model->forward(batch.data);
        Tensor<int> targets = batch.target.view({-1});
        loss = Tensor<float>::cross_entropy(output, targets);

        batch_loss += loss[{0}];
        int correct = 0;
        for (int i = 0; i < batch.data.size()[0]; i++)
        {
            Tensor<float> curr_batch = output[{{i, i + 1}}];
            Tensor<int> argmax = curr_batch.argmax();
            if (argmax[{0}] == targets[{i}])
            {
                correct++;
            }
        }
        batch_accuracy += (float)correct / batch.data.size()[0];
    }

    valid_loss = batch_loss / iters;
    valid_accuracy = batch_accuracy / iters;
}