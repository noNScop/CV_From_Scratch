#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include "vision_transforms.h"
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <torch/torch.h>
#include <unordered_map>
#include <variant>
#include <vector>

using TransformResult = std::variant<cv::Mat, torch::Tensor>;

struct Batch
{
    torch::Tensor data;
    torch::Tensor target;
};

class Dataset
{
  public:
    // Pure virtual method to get dataset size
    virtual size_t size() const = 0;

    // Pure virtual method to get a single data item at a specific index
    virtual std::pair<torch::Tensor, torch::Tensor> get_item(size_t index) const = 0;
};

// begin() must be called for auto shuffle, therefore it is preffered to use Dataloader like this:
// for(auto &batch : dataloader) {}
class DataLoader
{
  public:
    DataLoader(std::shared_ptr<Dataset> dataset, size_t batch_size, bool auto_shuffle = true);

    class Iterator
    {
      public:
        Iterator(DataLoader &dataloader, size_t index);

        Batch operator*();

        Iterator &operator++();

        bool operator!=(const Iterator &other) const;

      private:
        DataLoader &dataloader;
        size_t index;
    };

    Iterator begin();

    Iterator end();

    void shuffle();

  private:
    std::shared_ptr<Dataset> dataset;
    size_t batch_size;
    bool auto_shuffle;
    std::vector<size_t> indices;
    std::random_device rd;
    std::default_random_engine gen;
};

// Dataset in the simplest form, initialised with a vector of pairs of objects and targets
class BasicDataset : public Dataset
{
  public:
    BasicDataset(std::vector<std::pair<torch::Tensor, torch::Tensor>> data);

    size_t size() const override;

    std::pair<torch::Tensor, torch::Tensor> get_item(size_t index) const override;

  private:
    std::vector<std::pair<torch::Tensor, torch::Tensor>> data;
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
    ImageFolder(std::string path, std::shared_ptr<Transform> const &transform = nullptr);

    size_t size() const override;

    std::pair<torch::Tensor, torch::Tensor> get_item(size_t index) const override;

    std::unordered_map<std::string, int> class_to_idx;

  private:
    // a vector of file paths and labels
    std::vector<std::pair<cv::Mat, torch::Tensor>> data;
    std::shared_ptr<Transform> transform;
};

#endif