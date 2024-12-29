#include "data_utils.h"

using TransformResult = std::variant<cv::Mat, Tensor<float>>;

DataLoader::DataLoader(std::shared_ptr<Dataset> dataset, size_t batch_size, bool auto_shuffle, unsigned short num_workers)
    : dataset(std::move(dataset)), batch_size(batch_size), auto_shuffle(auto_shuffle), num_workers(num_workers), gen(rd())
{
    indices.resize(this->dataset->size());
    std::iota(indices.begin(), indices.end(), 0);
    last_batch_end_index = indices.size() + (indices.size() % batch_size);
}

DataLoader::Iterator::Iterator(DataLoader &dataloader, size_t index) : dataloader(dataloader), index(index)
{
}

Batch DataLoader::Iterator::operator*()
{
    // The last batch can be smaller than dataloader.batch_size
    size_t batch_size = std::min(dataloader.batch_size, dataloader.indices.size() - index);
    std::vector<Tensor<float>> data(batch_size);
    std::vector<Tensor<int>> target(batch_size);
    
    auto process_batch = [&](size_t start) {
        for (size_t i = start; i < batch_size; i += dataloader.num_workers)
        {
            auto [data_ten, target_ten] = dataloader.dataset->get_item(dataloader.indices[index + i]);
            data[i] = data_ten;
            target[i] = target_ten;
        }
    };

    std::vector<std::thread> threads;
    for (size_t i = 0; i < dataloader.num_workers; ++i)
    {
        threads.emplace_back(process_batch, i);
    }

    for (auto &thread : threads)
    {
        thread.join();
    }

    return {Tensor<float>::stack(data, 0), Tensor<int>::stack(target, 0)};
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
    return Iterator(*this, last_batch_end_index);
}

void DataLoader::shuffle()
{
    std::shuffle(indices.begin(), indices.end(), gen);
}

BasicDataset::BasicDataset(std::vector<std::pair<Tensor<float>, Tensor<int>>> data) : data(data)
{
}

size_t BasicDataset::size() const
{
    return data.size();
}

std::pair<Tensor<float>, Tensor<int>> BasicDataset::get_item(size_t index) const
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

ImageFolder::ImageFolder(std::string path, std::unordered_map<std::string, int> &class_to_idx,
                         std::shared_ptr<Transform> const &transform)
    : transform(transform), class_to_idx(class_to_idx)
{
    int label2idx = class_to_idx.size();
    std::unordered_set<std::string> class_names;
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
            std::string file_name = entry.path().filename().string();

            // Ignore files starting with a dot, e.g. .DS_Store - that would crash the program later
            if (file_name[0] == '.')
            {
                std::cout << "Ignoring hidden file: " << file_name << std::endl;
                continue;
            }

            int label;
            if (class_to_idx.count(class_name))
            {
                label = class_to_idx[class_name];
            }
            else
            {
                label = label2idx++;
                class_to_idx[class_name] = label;
            }

            Tensor<int> label_tensor = Tensor<int>({label});

            data.push_back({image_path, label_tensor});
        }
        else if (!entry.is_directory())
        {
            std::cerr << "Couldn't open a file: " << entry.path().string() << std::endl;
        }
    }
}

size_t ImageFolder::size() const
{
    return data.size();
}

std::pair<Tensor<float>, Tensor<int>> ImageFolder::get_item(size_t index) const
{
    if (index >= data.size())
    {
        throw std::out_of_range("Index out of range");
    }
    auto [image_path, label] = data[index];

    cv::Mat image = cv::imread(image_path, cv::IMREAD_UNCHANGED);

    TransformResult tensor = transform->apply(image);

    if (!std::holds_alternative<Tensor<float>>(tensor))
    {
        throw std::runtime_error("Transform must output a Tensor, non-Tensor object detected");
    }

    return {std::get<Tensor<float>>(tensor), label};
}
