#include "vision_transforms.h"
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <variant>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

class Dataset
{
  public:
    // Pure virtual method to get dataset size
    virtual size_t size() const = 0;

    // Pure virtual method to get a single data item at a specific index
    virtual std::pair<torch::Tensor, torch::Tensor> get_item(size_t index) const = 0;
};

class DataLoader
{
};

class ImageFolder : public Dataset
{
    // ----------REQUIRED DIRECTORY STRUCTURE----------
    // path/ <- train / validation dataset folder
    // class_1/ <- class name as folder name
    //     image01.jpeg
    //     image02.jpeg
    //     ...
    // class_2/
    //     image24.jpeg
    //     image25.jpeg
    //     ...
    // class_3/
    //     image37.jpeg
    //     ...
  public:
    ImageFolder(std::string path)
    {
        int label2idx = 0;
        for (const auto &entry : std::filesystem::recursive_directory_iterator(path))
        {
            if (entry.is_regular_file())
            {
                std::string image_path = entry.path().string();
                std::string class_name = entry.path().parent_path().filename().string();

                // Convert class name to label and store the maping
                int label = label2idx++;
                class_to_idx[class_name] = label;

                // Create label tensor
                torch::Tensor label_tensor = torch::tensor(label, torch::kInt64);

                // Store the image and label in the data vector
                data.push_back(std::make_pair(image_path, label_tensor));
            }
        }
    }

    size_t size() const override
    {
        return data.size();
    }

    std::pair<torch::Tensor, torch::Tensor> get_item(size_t index) const override
    {
        if (index < data.size())
        {
            // return data[index];
        }
        else
        {
            throw std::out_of_range("Index out of range");
        }
    }

    std::unordered_map<std::string, int> class_to_idx;

  private:
    // a vector of ile paths and labels
    std::vector<std::pair<std::string, torch::Tensor>> data;
};

int main()
{
    cv::Mat image = cv::imread("/Users/nonscop/Pictures/cropped-2560-1600-1105295.jpg");
    std::shared_ptr<Resize> resize1 = std::make_shared<Resize>(Resize(128, 128));
    std::shared_ptr<Resize> resize2 = std::make_shared<Resize>(Resize(3, 3));
    std::shared_ptr<ToTensor> totensor = std::make_shared<ToTensor>();
    Compose transforms = Compose{resize1, resize2, totensor};

    // cv::imshow("Display window", image);
    // cv::waitKey(0);
    // image = std::get<cv::Mat>(resize1(image));
    // cv::imshow("Display window", image);
    // cv::waitKey(0);
    // image = std::get<cv::Mat>(resize2(image));
    // cv::imshow("Display window", image);
    // cv::waitKey(0);

    torch::Tensor t = std::get<torch::Tensor>(transforms(image));
    std::cout << t << std::endl;

    // Wait for a key press indefinitely

    return 0;
}