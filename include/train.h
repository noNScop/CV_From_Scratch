#ifndef TRAIN_H
#define TRAIN_H

#include "data_utils.h"
#include "layers.h"
#include "models.h"
#include <iostream>
#include "tensor.h"

class Optimizer
{
  public:
    Optimizer(std::vector<Tensor<float>> parameters, float learning_rate);

    float learning_rate;

    void zero_grad();

    virtual void step() = 0;

  protected:
    std::vector<Tensor<float>> parameters;
};

class SGD : public Optimizer
{
  public:
    SGD(std::vector<Tensor<float>> parameters, float learning_rate = 0.01, float momentum = 0.9);

    void step() override;

  private:
    float momentum;
    std::vector<Tensor<float>> ema; // exponential moving average
};

class Learner
{
  public:
    Learner(std::shared_ptr<Module> model);

    void train(DataLoader &train_dl, DataLoader &valid_dl, int epochs);

  private:
    std::shared_ptr<Module> model;
    float train_loss;
    float train_accuracy;
    float valid_loss;
    float valid_accuracy;
    int train_iters;
    int valid_iters;

    Tensor<float> output;
    Tensor<float> loss;

    void train_step(DataLoader &train_dl, SGD &optimizer);

    void valid_step(DataLoader &valid_dl);
};

#endif