#include "data_utils.h"
#include "layers.h"
#include "models.h"
#include "train.h"
#include "vision_transforms.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <fstream>
#include <iostream>

#ifndef DATA_DIR
#define DATA_DIR "./data" // Fallback if DATA_DIR is not defined
#endif

int main()
{
    std::shared_ptr<MnistCNN> model = std::make_shared<MnistCNN>(MnistCNN());
    // Initialize dataset by providing the path to the MNIST data directory
    std::unordered_map<std::string, int> class_to_idx;
    // Adjust this path if necessary
    std::string mnist_train_path = std::string(DATA_DIR) + "/training";
    // Adjust this path if necessary
    std::string mnist_valid_path = std::string(DATA_DIR) + "/testing";
    // Create datasets
    std::shared_ptr<ImageFolder> train_ds = std::make_shared<ImageFolder>(mnist_train_path, class_to_idx);
    std::shared_ptr<ImageFolder> valid_ds = std::make_shared<ImageFolder>(mnist_valid_path, class_to_idx);

    // Create the reverse mapping for inference
    std::unordered_map<int, std::string> idx_to_class;
    for (auto const &pair : class_to_idx)
    {
        idx_to_class[pair.second] = pair.first;
    }

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

        if (std::cin.fail())
        {
            std::cin.clear();                                                   // Clear the error state
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Ignore the invalid input
            std::cout << "\nInvalid choice. Please select a valid option.\n";
            continue;
        }

        switch (choice)
        {
        case 1: // Training
        {
            int epochs;

            // Loop until valid input for epochs is provided
            while (true)
            {
                std::string input;
                std::cout << "\nEnter the number of epochs: ";
                std::cin >> input;

                // Check if the input is valid
                try
                {
                    epochs = std::stoi(input);
                    if (epochs <= 0)
                    {
                        throw std::invalid_argument("Non-positive integer");
                    }
                    break; // Exit loop if input is valid
                }
                catch (const std::invalid_argument &e)
                {
                    std::cin.clear();                                                   // Clear error flag
                    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Discard invalid input
                    std::cerr << "Invalid input. Please enter a positive integer.\n";
                }
            }
            // Create DataLoaders
            DataLoader train_dl = DataLoader(train_ds, 32);
            DataLoader valid_dl = DataLoader(valid_ds, 32, false);

            Learner learn = Learner(model);
            learn.train(train_dl, valid_dl, epochs);
            break;
        }
        case 2: // Saving weights
            {
                std::string save_path;
                std::cout << "Enter the path to save the model: ";
                std::cin >> save_path;
                std::ofstream file(save_path, std::ios::binary); 
                if (!file.is_open())
                {
                    throw std::ios_base::failure("Failed to open file for saving.");
                }
                cereal::BinaryOutputArchive archive_save(file);
                archive_save(model);
                std::cout << "Saved the model to: " << save_path << "\n";
            }
            break;
        case 3: // Loading weights
            {
                std::string load_path;
                std::cout << "Enter the path to load the model: ";
                std::cin >> load_path;
                std::ifstream file(load_path, std::ios::binary);
                if (!file.is_open())
                {
                    throw std::ios_base::failure("Failed to open file for loading.");
                }
                cereal::BinaryInputArchive archive(file); 
                archive(model);
                std::cout << "Loaded the model from: " << load_path << "\n";
            }
            break;
        case 4: // Inference
        {
            cv::Mat image;
            std::string path;

            std::cout << "Provide a path to image: ";
            std::cin >> path;

            if (!std::filesystem::exists(path))
            {
                std::cerr << "The file at path '" << path << "' does not exist. Please try again." << std::endl;
                break;
            }

            image = cv::imread(path, cv::IMREAD_GRAYSCALE);
            if (image.empty())
            {
                std::cerr << "Could not open or find the image at path: " << path << std::endl;
                break;
            }

            std::shared_ptr<Resize> resize = std::make_shared<Resize>(28, 28);
            std::shared_ptr<ToTensor> totensor = std::make_shared<ToTensor>();
            Compose transform({resize, totensor});

            Tensor<float> img_tensor = std::get<Tensor<float>>(transform(image)).view({1, 1, 28, 28});

            model->set_training(false);
            NoGradGuard no_grad;
            auto output = model->forward(img_tensor);
            Tensor<float> curr_batch = output[{{0, 1}}];
            Tensor<int> argmax = curr_batch.argmax();
            std::string prediction = idx_to_class[argmax[{0}]];
            Tensor<float> softmax_out = Tensor<float>::softmax(output, 1);
            auto confidence = softmax_out.max();

            std::string output_message =
                "\nPrediction: " + prediction + ", Confidence: " + std::to_string(confidence[{0}]);

            std::cout << output_message << std::endl;
            break;
        }
        case 5:
            std::cout << "Exiting..." << std::endl;
            return 0;
        default:
            std::cout << "Invalid choice. Please select a valid option." << std::endl;
            break;
        }
    }
}