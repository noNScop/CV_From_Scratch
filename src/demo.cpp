#include "train.h"
#include "layers.h"
#include "models.h"
#include <torch/torch.h>

int main()
{
    // Initialize dataset by providing the path to the MNIST data directory
    std::string mnist_data_path = "/home/vertex/Desktop/CV_From_Scratch/data"; // Adjust this path if necessary
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
    learn.train(train_dl, valid_dl, 1);

    return 0;
}