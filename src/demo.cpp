#include "layers.h"
#include "models.h"
#include "train.h"
#include <iostream>
#include <torch/torch.h>



int main()
{
    int choice;
    while (true)
    {
        std::cout << "\nSelect an option:\n";
        std::cout << "[1. Train CNN on MNIST dataset]" << std::endl;
        std::cout << "[2. Save weights to memory]" << std::endl;
        std::cout << "[3. Load weights from memory]" << std::endl;
        std::cout << "[4. Test on inference]" << std::endl;
        std::cout << "[5. Exit]" << std::endl;
        std::cin >> choice;

        switch (choice)
        {
        case 1:
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
            learn.train(train_dl, valid_dl, 5);
            break;
        }
        case 2:
            std::cout << "Not implemented yet" << std::endl;
            break;
        case 3:
            std::cout << "Not implemented yet" << std::endl;
            break;
        case 4:
            std::cout << "Not implemented yet" << std::endl;
            break;
        case 5:
            std::cout << "Exiting..." << std::endl;
            return 0;
        default:
            std::cout << "Invalid choice. Please select a valid option." << std::endl;
            break;
        }
    }
}