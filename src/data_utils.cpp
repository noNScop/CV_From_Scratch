#include "data_utils.h"

using TransformResult = std::variant<cv::Mat, torch::Tensor>;

DataLoader::DataLoader(std::shared_ptr<Dataset> dataset, size_t batch_size, bool auto_shuffle)
    : dataset(std::move(dataset)), batch_size(batch_size), auto_shuffle(auto_shuffle), gen(rd())
{
    // Initialize indices
    indices.resize(this->dataset->size());
    std::iota(indices.begin(), indices.end(), 0);
}

DataLoader::Iterator::Iterator(DataLoader &dataloader, size_t index) : dataloader(dataloader), index(index)
{
}

Batch DataLoader::Iterator::operator*()
{
    std::vector<torch::Tensor> data;
    std::vector<torch::Tensor> target;
    for (size_t i = 0; i < dataloader.batch_size && index + i < dataloader.indices.size(); ++i)
    {
        auto [data_ten, target_ten] = dataloader.dataset->get_item(dataloader.indices[index + i]);
        data.push_back(data_ten);
        target.push_back(target_ten);
    }
    return {torch::stack(data, 0), torch::stack(target, 0)};
}

DataLoader::Iterator &DataLoader::Iterator::operator++()
{
    index += dataloader.batch_size;
    return *this;
}

bool DataLoader::Iterator::operator!=(const Iterator &other) const
{
    return index != other.index;
}

DataLoader::Iterator DataLoader::begin()
{
    if (auto_shuffle)
    {
        shuffle();
    }
    return Iterator(*this, 0);
}

DataLoader::Iterator DataLoader::end()
{
    return Iterator(*this, indices.size());
}

void DataLoader::shuffle()
{
    std::shuffle(indices.begin(), indices.end(), gen);
}

BasicDataset::BasicDataset(std::vector<std::pair<torch::Tensor, torch::Tensor>> data) : data(data)
{
}

size_t BasicDataset::size() const
{
    return data.size();
}

std::pair<torch::Tensor, torch::Tensor> BasicDataset::get_item(size_t index) const
{
    if (index < data.size())
    {
        return data[index];
    }
    else
    {
        throw std::out_of_range("Index out of range");
    }
}

ImageFolder::ImageFolder(std::string path, std::shared_ptr<Transform> const &transform) : transform(transform)
{
    int label2idx = 0;
    if (!transform)
    {
        this->transform = std::make_shared<ToTensor>();
    }
    for (const auto &entry : std::filesystem::recursive_directory_iterator(path))
    {
        if (entry.is_regular_file())
        {
            std::string image_path = entry.path().string();
            std::string class_name = entry.path().parent_path().filename().string();

            // Convert class name to label and store the mapping
            int label = label2idx++;
            class_to_idx[class_name] = label;

            // Create label tensor
            torch::Tensor label_tensor = torch::tensor(label, torch::kInt64);

            cv::Mat image = cv::imread(image_path);
            data.push_back({image, label_tensor});
        }
        else
        {
            std::cerr << "Couldn't open a file: " << entry.path().string() << std::endl;
        }
    }
}

size_t ImageFolder::size() const
{
    return data.size();
}

std::pair<torch::Tensor, torch::Tensor> ImageFolder::get_item(size_t index) const
{
    if (index >= data.size())
    {
        throw std::out_of_range("Index out of range");
    }

    auto [image, label] = data[index];
    TransformResult tensor = transform->apply(image);

    if (!std::holds_alternative<torch::Tensor>(tensor))
    {
        throw std::runtime_error("Transform must output a Tensor, non-Tensor object detected");
    }

    return {std::get<torch::Tensor>(tensor), label};
}
