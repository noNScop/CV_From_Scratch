#include "layers.h"
#include "models.h"
#include <iostream>
#include <torch/torch.h>

class SGD
{
};

class Learner
{
  public:
    Learner(std::shared_ptr<Module> model) : model(model)
    {
    }

    template <typename DataLoader> void train(DataLoader &train_dl, DataLoader &valid_dl, int epochs)
    {
        torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.01).momentum(0.9));
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

    template <typename DataLoader> void train_step(DataLoader &train_dl, torch::optim::SGD &optimizer)
    {
        namespace F = torch::nn::functional;

        float avg_accuracy = 0;
        float avg_loss = 0;
        int iters = 0;
        model->set_training(true);

        for (auto &batch : *train_dl)
        {
            ++iters;
            output = (*model)(batch.data);
            loss = F::cross_entropy(output, batch.target);

            loss.backward();
            optimizer.step();
            optimizer.zero_grad();

            avg_loss += loss.item<float>();
            avg_accuracy += (output.argmax(1) == batch.target).sum().template item<float>() / batch.data.size(0);
        }

        train_loss += avg_loss / iters;
        train_accuracy += avg_accuracy / iters;
    }

    template <typename DataLoader> void valid_step(DataLoader &valid_dl)
    {
        namespace F = torch::nn::functional;

        torch::NoGradGuard nograd;
        float avg_accuracy = 0;
        float avg_loss = 0;
        int iters = 0;
        model->set_training(false);

        for (auto &batch : *valid_dl)
        {
            ++iters;
            output = (*model)(batch.data);
            loss = F::cross_entropy(output, batch.target);

            avg_loss += loss.item<float>();
            avg_accuracy += (output.argmax(1) == batch.target).sum().template item<float>() / batch.data.size(0);
        }

        valid_loss += avg_loss / iters;
        valid_accuracy += avg_accuracy / iters;
    }
};

int main()
{
    // Initialize dataset by providing the path to the MNIST data directory
    std::string mnist_data_path = "/Users/nonscop/Desktop/CV_From_Scratch/data"; // Adjust this path if necessary
    torch::data::datasets::MNIST train_dsa(mnist_data_path, torch::data::datasets::MNIST::Mode::kTrain);
    auto train_ds = train_dsa.map(torch::data::transforms::Stack<>());
    torch::data::datasets::MNIST valid_dsa(mnist_data_path, torch::data::datasets::MNIST::Mode::kTest);
    auto valid_ds = valid_dsa.map(torch::data::transforms::Stack<>());

    // Create a DataLoader
    auto train_dl = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_ds), 32);
    auto valid_dl = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(valid_ds), 32);
    // Iterate through batches
    std::shared_ptr<MnistCNN> model = std::make_shared<MnistCNN>(MnistCNN());
    Learner learn = Learner(model);
    learn.train(train_dl, valid_dl, 5);

    return 0;
}