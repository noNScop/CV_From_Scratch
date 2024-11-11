#include <catch2/catch_test_macros.hpp>
#include "layers.h"
#include "vision_transforms.h"
#include "models.h"
#include "train.h"
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

TEST_CASE( "Linear Layer", "[layers]" ) 
{   
    // Different initializations
    REQUIRE_NOTHROW( Linear(1, 1, true, true) );
    REQUIRE_NOTHROW( Linear(1, 1, true, false) );
    REQUIRE_NOTHROW( Linear(1, 1, false, true) );
    REQUIRE_NOTHROW( Linear(1, 1, false, false) );

    // Test parameters registering
    Linear linear = Linear(3, 6);
    REQUIRE( linear.parameters()[0].equal(linear.weights) );
    REQUIRE( linear.parameters()[1].equal(linear.bias) );

    // Test output sizes
    torch::Tensor sample_input = torch::randn({32, 3});
    REQUIRE( linear(sample_input).sizes() ==  torch::IntArrayRef({32, 6}) );

    // Test output values
    linear.weights = torch::tensor({1, 2, 3, 4, 5, 6, 7, 8, 9}).view({3, 3});
    linear.bias = torch::tensor({0, 0, 0});
    torch::Tensor test_input = torch::tensor({1, 2, 3, 4, 5, 6}).view({2, 3});
    torch::Tensor test_output = torch::tensor({14, 32, 50, 32, 77, 122}).view({2, 3});
    REQUIRE( linear(test_input).equal(test_output) );
}

TEST_CASE( "Convolutional layer", "[layers]" )
{
    // Different initializations
    REQUIRE_NOTHROW( Conv2d(1, 1, 1, 1, 0, true, true) );
    REQUIRE_NOTHROW( Conv2d(1, 1, 1, 1, 0, true, false) );
    REQUIRE_NOTHROW( Conv2d(1, 1, 1, 1, 0, false, true) );
    REQUIRE_NOTHROW( Conv2d(1, 1, 1, 1, 0, false, false) );

    // Test parameters registering
    Conv2d conv = Conv2d(3, 10, 3);
    REQUIRE( conv.parameters()[0].equal(conv.weights) );
    REQUIRE( conv.parameters()[1].equal(conv.bias) );

    // Test output sizes
    torch::Tensor sample_input = torch::randn({32, 3, 224, 224});
    REQUIRE( conv(sample_input).sizes() == torch::IntArrayRef({32, 10, 222, 222}) );

    // Test output values
    conv = Conv2d(1, 1, 2, 1, 0, false, false);
    conv.weights = torch::tensor({2., 1., 1., -1.}).view({1, 1, 2, 2});
    torch::Tensor test_input = torch::tensor({1., 2., 3., 4., 5., 6., 7., 8., 9.}).view({1, 1, 3, 3});
    torch::Tensor test_output = torch::tensor({3., 6., 12., 15.}).view({1, 1, 2, 2});
    REQUIRE( conv(test_input).equal(test_output) );
}

TEST_CASE( "Batch normalization layer", "[layers]" )
{
    // Different initializations
    REQUIRE_NOTHROW( BatchNorm2d(1, true, true) );
    REQUIRE_NOTHROW( BatchNorm2d(1, true, false) );
    REQUIRE_NOTHROW( BatchNorm2d(1, false, true) );
    REQUIRE_NOTHROW( BatchNorm2d(1, false, false) );

    // Test parameters registering
    BatchNorm2d batch_norm = BatchNorm2d(3, true, true);
    REQUIRE( batch_norm.parameters()[0].equal(torch::zeros({3})) );
    batch_norm = BatchNorm2d(3);
    REQUIRE( batch_norm.parameters()[0].equal(torch::ones({3})) );
    REQUIRE( batch_norm.parameters()[1].equal(torch::zeros({3})) );

    // Test normalization on training
    torch::Tensor sample_input = torch::rand({1, 3, 10, 10});
    torch::Tensor normalized_input = batch_norm(sample_input);

    // Mean
    float difference = (torch::zeros({3}) - normalized_input.mean({0, 2, 3})).mean().abs().item<float>();
    REQUIRE( difference < 0.01 );

    // Variance
    difference = (torch::ones({3}) - normalized_input.var({0, 2, 3})).mean().abs().item<float>();
    REQUIRE( difference < 0.01 );

    // Test normalization on validation - check running stats
    for (int i = 0; i < 1000; ++i)
    {
        normalized_input = batch_norm(sample_input);
    }
    batch_norm.set_training(false);
    normalized_input = batch_norm(sample_input);

    // Mean
    difference = (torch::zeros({3}) - normalized_input.mean({0, 2, 3})).mean().abs().item<float>();
    REQUIRE( difference < 0.01 );

    // Variance
    difference = (torch::ones({3}) - normalized_input.var({0, 2, 3})).mean().abs().item<float>();
    REQUIRE( difference < 0.01 );
}

TEST_CASE( "Sequential layer", "[layers]" )
{
    std::shared_ptr<Linear> linear1 = std::make_shared<Linear>(Linear(3, 6));
    std::shared_ptr<Linear> linear2 = std::make_shared<Linear>(Linear(6, 12));
    Sequential seq = Sequential({linear1, linear2});
    torch::Tensor sample_input = torch::randn({32, 3});
    torch::Tensor test_output = sample_input;
    for (auto layer : seq.get_children())
    {
        test_output = layer->forward(test_output);
    }

    // Test functionality / Module registration
    REQUIRE( seq(sample_input).equal(test_output) );
}

TEST_CASE( "ReLU layer", "[layers]" )
{
    torch::Tensor x = torch::tensor({-1, 1});
    ReLU relu = ReLU();
    REQUIRE( relu(x).equal(torch::tensor({0, 1})) );
}

TEST_CASE( "SGD optimizer", "[optimizers]" )
{
    torch::Tensor x = torch::rand({10, 1});
    torch::Tensor y = 3 * x + 2;
    Linear linear = Linear(1, 1);

    // Different initializations
    REQUIRE_NOTHROW( SGD(linear.parameters()) );
    REQUIRE_NOTHROW( SGD(linear.parameters(), 1e-4, 0.99) );

    SGD optimizer = SGD(linear.parameters());
    torch::Tensor out;
    torch::Tensor loss;

    // Test on simple linear regression task
    for (int i = 0; i < 1000; ++i)
    {
        out = linear(x);
        loss = (out - y).pow(2).mean();
        loss.backward();
        optimizer.step();
        optimizer.zero_grad();
    }

    REQUIRE( (linear.weights - 3).abs().item<float>() < 0.01 );
    REQUIRE( (linear.bias - 2).abs().item<float>() < 0.01 );
}

TEST_CASE( "Compose, Resize and ToTensor transfroms", "[Transforms]" )
{
    cv::Mat test_image(224, 320, CV_32F, cv::Scalar(0));
    std::shared_ptr<Resize> resize = std::make_shared<Resize>(Resize(3, 3));
    std::shared_ptr<ToTensor> totensor = std::make_shared<ToTensor>();
    Compose transforms = Compose{resize, totensor};

    torch::Tensor t = std::get<torch::Tensor>(transforms(test_image));
    REQUIRE( t.sizes() == torch::IntArrayRef({1, 3, 3}) );
}