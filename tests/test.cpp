#include "data_utils.h"
#include "layers.h"
#include "models.h"
#include "train.h"
#include "vision_transforms.h"
#include <catch2/catch_test_macros.hpp>
#include <opencv2/opencv.hpp>
#include "tensor.h"

TEST_CASE("Linear Layer", "[layers]")
{
    REQUIRE_NOTHROW(Linear(1, 1, true, true));
    REQUIRE_NOTHROW(Linear(1, 1, true, false));
    REQUIRE_NOTHROW(Linear(1, 1, false, true));
    REQUIRE_NOTHROW(Linear(1, 1, false, false));

    Linear linear = Linear(3, 6);
    REQUIRE(linear.parameters()[0].equal(linear.weights));
    REQUIRE(linear.parameters()[1].equal(linear.bias));

    Tensor<float> sample_input = Tensor<float>::randn({32, 3});
    REQUIRE(linear(sample_input).size() == std::vector<size_t>({32, 6}));

    linear.weights = Tensor<float>({1, 2, 3, 4, 5, 6, 7, 8, 9}).view({3, 3});
    linear.bias = Tensor<float>({0, 0, 0});
    Tensor<float> test_input = Tensor<float>({1, 2, 3, 4, 5, 6}).view({2, 3});
    Tensor<float> test_output = Tensor<float>({14, 32, 50, 32, 77, 122}).view({2, 3});
    REQUIRE(linear(test_input).equal(test_output));
}

TEST_CASE("Convolutional layer", "[layers]")
{
    REQUIRE_NOTHROW(Conv2d(1, 1, 1, 1, 0, true, true));
    REQUIRE_NOTHROW(Conv2d(1, 1, 1, 1, 0, true, false));
    REQUIRE_NOTHROW(Conv2d(1, 1, 1, 1, 0, false, true));
    REQUIRE_NOTHROW(Conv2d(1, 1, 1, 1, 0, false, false));

    Conv2d conv = Conv2d(3, 10, 3);
    REQUIRE(conv.parameters()[0].equal(conv.weights));
    REQUIRE(conv.parameters()[1].equal(conv.bias));

    Tensor<float> sample_input = Tensor<float>::randn({32, 3, 224, 224});
    REQUIRE(conv(sample_input).size() == std::vector<size_t>({32, 10, 222, 222}));

    conv = Conv2d(1, 1, 2, 1, 0, false, false);
    conv.weights = Tensor<float>({2., 1., 1., -1.}).view({1, 1, 2, 2});
    Tensor<float> test_input = Tensor<float>({1., 2., 3., 4., 5., 6., 7., 8., 9.}).view({1, 1, 3, 3});
    Tensor<float> test_output = Tensor<float>({3., 6., 12., 15.}).view({1, 1, 2, 2});
    REQUIRE(conv(test_input).equal(test_output));
}

TEST_CASE("Batch normalization layer", "[layers]")
{
    REQUIRE_NOTHROW(BatchNorm2d(1, true, true));
    REQUIRE_NOTHROW(BatchNorm2d(1, true, false));
    REQUIRE_NOTHROW(BatchNorm2d(1, false, true));
    REQUIRE_NOTHROW(BatchNorm2d(1, false, false));

    BatchNorm2d batch_norm = BatchNorm2d(3, true, true);
    
    Tensor<float> three_zeros = Tensor<float>::zeros({3});
    Tensor<float> three_ones = Tensor<float>::ones({3});
    
    REQUIRE(batch_norm.parameters()[0].equal(three_zeros));
    
    batch_norm = BatchNorm2d(3);
    REQUIRE(batch_norm.parameters()[0].equal(three_ones));
    REQUIRE(batch_norm.parameters()[1].equal(three_zeros));

    Tensor<float> sample_input = Tensor<float>::randn({1, 3, 10, 10});
    Tensor<float> normalized_input = batch_norm(sample_input);

    Tensor<float> normalized_input_mean = normalized_input.mean({0, 2, 3}, false);
    Tensor<float> difference = (Tensor<float>::zeros({3}) - normalized_input_mean);
    Tensor<float> diff_mean = difference.mean();
    REQUIRE(std::abs(diff_mean[{0}]) < 0.01);

    Tensor<float> normalized_input_var = normalized_input.var({0, 2, 3}, false);
    difference = (Tensor<float>::ones({3}) - normalized_input_var);
    diff_mean = difference.mean();
    REQUIRE(std::abs(diff_mean[{0}]) < 0.01);

    for (int i = 0; i < 1000; ++i)
    {
        normalized_input = batch_norm(sample_input);
    }
    batch_norm.set_training(false);
    normalized_input = batch_norm(sample_input);

    normalized_input_mean = normalized_input.mean({0, 2, 3}, false);
    difference = (Tensor<float>::zeros({3}) - normalized_input_mean);
    diff_mean = difference.mean();
    REQUIRE(std::abs(diff_mean[{0}]) < 0.01);

    normalized_input_var = normalized_input.var({0, 2, 3}, false);
    difference = (Tensor<float>::ones({3}) - normalized_input_var);
    diff_mean = difference.mean();
    REQUIRE(std::abs(diff_mean[{0}]) < 0.01);
}

TEST_CASE("Sequential layer", "[layers]")
{
    std::shared_ptr<Linear> linear1 = std::make_shared<Linear>(Linear(3, 6));
    std::shared_ptr<Linear> linear2 = std::make_shared<Linear>(Linear(6, 12));
    Sequential seq = Sequential({linear1, linear2});
    Tensor<float> sample_input = Tensor<float>::randn({32, 3});
    Tensor<float> test_output = sample_input;
    for (auto layer : seq.get_children())
    {
        test_output = layer->forward(test_output);
    }

    REQUIRE(seq(sample_input).equal(test_output));
}

TEST_CASE("ReLU layer", "[layers]")
{
    Tensor<float> x = Tensor<float>({-1, 1});
    ReLU relu = ReLU();
    Tensor<float> expected_result = Tensor<float>({0, 1});
    REQUIRE(relu(x).equal(expected_result));
}

TEST_CASE("SGD optimizer", "[optimizers]")
{
    Tensor<float> x = Tensor<float>::randn({10, 1});
    Tensor<float> y = (x * 3) + 2;
    Linear linear = Linear(1, 1);

    REQUIRE_NOTHROW(SGD(linear.parameters()));
    REQUIRE_NOTHROW(SGD(linear.parameters(), 1e-4, 0.99));

    SGD optimizer = SGD(linear.parameters());
    Tensor<float> out;
    Tensor<float> loss;

    for (int i = 0; i < 100; ++i)
    {
        out = linear(x);
        loss = (out - y).pow(2).mean();
        loss.backward();
        optimizer.step();
        optimizer.zero_grad();
    }
    Tensor<float> weights_normalized = linear.weights - 3;
    Tensor<float> bias_normalized = linear.bias - 2;
    REQUIRE(std::abs(weights_normalized[{0, 0}]) < 0.02);
    REQUIRE(std::abs(bias_normalized[{0}]) < 0.02);
}

TEST_CASE("Compose, Resize and ToTensor transfroms", "[transforms]")
{
    cv::Mat test_image(224, 320, CV_32F, cv::Scalar(0));
    std::shared_ptr<Resize> resize = std::make_shared<Resize>(Resize(3, 3));
    std::shared_ptr<ToTensor> totensor = std::make_shared<ToTensor>();
    Compose transforms = Compose{resize, totensor};

    Tensor<float> t = std::get<Tensor<float>>(transforms(test_image));
    REQUIRE(t.size() == std::vector<size_t>({1, 3, 3}));
}

TEST_CASE("DataLoader batching", "[dataloader]")
{
    std::vector<std::pair<Tensor<float>, Tensor<int>>> data;
    for (int i = 0; i < 8; ++i)
    {
        Tensor<float> tensor_data = Tensor<float>::zeros({3});
        Tensor<int> targets = Tensor<int>::zeros({1});
        data.push_back({tensor_data, targets});
    }

    std::shared_ptr<BasicDataset> dataset = std::make_shared<BasicDataset>(data);
    DataLoader dataloader = DataLoader(dataset, 2, false);

    int number_of_batches = 0;
    for (auto batch : dataloader)
    {
        REQUIRE(batch.data.size() == std::vector<size_t>({2, 3}));
        REQUIRE(batch.target.size() == std::vector<size_t>({2, 1}));
        ++number_of_batches;
    }

    REQUIRE(number_of_batches == 4);
}

TEST_CASE("DataLoader shuffling", "[dataloader]")
{
    std::vector<std::pair<Tensor<float>, Tensor<int>>> data;
    for (int i = 0; i < 100; ++i)
    {
        Tensor<float> tensor_data = Tensor<float>({static_cast<float>(i), static_cast<float>(i + 1)});
        Tensor<int> targets = Tensor<int>::zeros({1});
        data.push_back({tensor_data, targets});
    }

    std::shared_ptr<BasicDataset> dataset = std::make_shared<BasicDataset>(data);
    DataLoader dataloader = DataLoader(dataset, 2);

    std::vector<Tensor<float>> first_cycle;
    for (auto batch : dataloader)
    {
        first_cycle.push_back(batch.data);
    }

    int idx = 0;
    bool flag = false;

    for (auto batch : dataloader)
    {
        if (!first_cycle[idx++].equal(batch.data))
        {
            flag = true;
            break;
        }
    }

    REQUIRE(flag);
}