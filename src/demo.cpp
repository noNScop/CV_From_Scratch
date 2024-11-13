#include "layers.h"
#include "models.h"
#include "train.h"
#include "vision_transforms.h"
#include "data_utils.h"
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
        case 1: {
            // Initialize dataset by providing the path to the MNIST data directory
            // Adjust this path if necessary
            std::string mnist_train_path = "/Users/nonscop/Desktop/CV_From_Scratch/data/training";
            // Adjust this path if necessary
            std::string mnist_valid_path = "/Users/nonscop/Desktop/CV_From_Scratch/data/testing";

            // Create datasets
            std::shared_ptr<ImageFolder> train_ds = std::make_shared<ImageFolder>(mnist_train_path);
            std::shared_ptr<ImageFolder> valid_ds = std::make_shared<ImageFolder>(mnist_valid_path);
            // Create DataLoaders
            DataLoader train_dl = DataLoader(train_ds, 32);
            DataLoader valid_dl = DataLoader(valid_ds, 32, false);

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