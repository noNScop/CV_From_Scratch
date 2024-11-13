#ifndef TRAIN_H
#define TRAIN_H

#include "layers.h"
#include "models.h"
#include <iostream>
#include <torch/torch.h>

class Optimizer
{
  public:
    Optimizer(std::vector<torch::Tensor> parameters, float learning_rate);

    float learning_rate;

    void zero_grad();

    virtual void step() = 0;

  protected:
    std::vector<torch::Tensor> parameters;
};

class SGD : public Optimizer
{
  public:
    SGD(std::vector<torch::Tensor> parameters, float learning_rate = 0.01, float momentum = 0.9);

    void step() override;

  private:
    float momentum;
    std::vector<torch::Tensor> ema; // exponential moving average
};

class Learner
{
  public:
    Learner(std::shared_ptr<Module> model);

    template <typename DataLoader> void train(DataLoader &train_dl, DataLoader &valid_dl, int epochs);

  private:
    std::shared_ptr<Module> model;
    float train_loss;
    float train_accuracy;
    float valid_loss;
    float valid_accuracy;
    int train_iters;
    int valid_iters;

    torch::Tensor output;
    torch::Tensor loss;

    template <typename DataLoader> void train_step(DataLoader &train_dl, SGD &optimizer);

    template <typename DataLoader> void valid_step(DataLoader &valid_dl);
};

template <typename DataLoader> void Learner::train(DataLoader &train_dl, DataLoader &valid_dl, int epochs)
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

template <typename DataLoader> void Learner::train_step(DataLoader &train_dl, SGD &optimizer)
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

template <typename DataLoader> void Learner::valid_step(DataLoader &valid_dl)
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

#endif