#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include "vision_transforms.h"
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <thread>
#include "tensor.h"
#include <unordered_map>
#include <variant>
#include <vector>

using TransformResult = std::variant<cv::Mat, Tensor<float>>;

struct Batch
{
    Tensor<float> data;
    Tensor<int> target;
};

class Dataset
{
  public:
    // Pure virtual method to get dataset size
    virtual size_t size() const = 0;

    // Pure virtual method to get a single data item at a specific index
    virtual std::pair<Tensor<float>, Tensor<int>> get_item(size_t index) const = 0;
};

// begin() must be called for auto shuffle, therefore it is preffered to use Dataloader like this:
// for(auto &batch : dataloader) {}
class DataLoader
{
  public:
    DataLoader(std::shared_ptr<Dataset> dataset, size_t batch_size, bool auto_shuffle = true, unsigned short num_workers = 10);

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

    // shuffles the indices vector
    void shuffle();

  private:
    std::shared_ptr<Dataset> dataset;
    size_t batch_size;
    bool auto_shuffle;
    unsigned short num_workers;
    std::vector<size_t> indices;
    // actually last batch end index + 1, serves as the ending condition in Iterator of Dataloader
    size_t last_batch_end_index;
    std::random_device rd;
    std::default_random_engine gen;
};

// Dataset in the simplest form, initialised with a vector of pairs of objects and targets
class BasicDataset : public Dataset
{
  public:
    BasicDataset(std::vector<std::pair<Tensor<float>, Tensor<int>>> data);

    size_t size() const override;

    std::pair<Tensor<float>, Tensor<int>> get_item(size_t index) const override;

  private:
    std::vector<std::pair<Tensor<float>, Tensor<int>>> data;
};

// In this version it is processing and transforming images in get_item method, which is more STABLE but SLOWER
// than doing it for all images during initialisation of ImageFolder Dataset, for FASTER PERFORMANCE on MNIST dataset
// checkout to "final code" commit, however it is LESS STABLE and sometimes segmentation fault occurs during
// initialisation of ImageFolder Dataset
class ImageFolder : public Dataset
{
    // ----------REQUIRED DIRECTORY STRUCTURE----------
    // path/ <- train / validation dataset folder
    // class_1/ <- class name as folder name
    //     image01.jpeg <- not necessarily .jpeg
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
    ImageFolder(std::string path, std::unordered_map<std::string, int> &class_to_idx,
                std::shared_ptr<Transform> const &transform = nullptr);

    size_t size() const override;

    std::pair<Tensor<float>, Tensor<int>> get_item(size_t index) const override;

    std::unordered_map<std::string, int> &class_to_idx;

  private:
    // a vector of data paths and targets
    std::vector<std::pair<std::string, Tensor<int>>> data;
    std::shared_ptr<Transform> transform;
};

#endif