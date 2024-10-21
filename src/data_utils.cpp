#include <iostream>
#include <torch/torch.h>

int main()
{
    // Initialize dataset by providing the path to the MNIST data directory
    std::string mnist_data_path = "/home/vertex/Desktop/CV_From_Scratch/data"; // Adjust this path if necessary
    torch::data::datasets::MNIST train_ds(mnist_data_path, torch::data::datasets::MNIST::Mode::kTrain);
    torch::data::datasets::MNIST test_ds(mnist_data_path, torch::data::datasets::MNIST::Mode::kTest);

    std::cout << train_ds.size().value() << std::endl;
    std::cout << test_ds.size().value() << std::endl;

    // Create a DataLoader
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_ds), 32);
    // Iterate through batches
    torch::data::transforms::Stack stack;
    for (auto &batch : *data_loader)
    {   
        auto images = stack.apply_batch(batch).data;
        auto labels = stack.apply_batch(batch).target;

        // Check the sizes of the images and labels
        std::cout << "Batch size: " << images.size(0) << std::endl;         // Number of images in the batch
        std::cout << "Image tensor size: " << images.sizes() << std::endl;  // Size of the image tensor
        std::cout << "Labels tensor size: " << labels.sizes() << std::endl; // Size of the labels tensor

        // Here you can add code to process the images and labels, e.g., passing them through a model
    }

    return 0;
}
