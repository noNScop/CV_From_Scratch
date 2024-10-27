#include <catch2/catch_test_macros.hpp>
#include "train.h"
#include "layers.h"
#include "models.h"
#include <vector>
#include <torch/torch.h>

TEST_CASE( "Linear Layer", "[layers]" ) 
{   
    // Different inicializations
    REQUIRE_NOTHROW( Linear(1, 1, true, true) );
    REQUIRE_NOTHROW( Linear(1, 1, true, false) );
    REQUIRE_NOTHROW( Linear(1, 1, false, true) );
    REQUIRE_NOTHROW( Linear(1, 1, false, false) );

    // Test parameters registering
    Linear linear = Linear(3, 6);
    REQUIRE( linear.parameters()[0].equal(linear.weights) );
    REQUIRE( linear.parameters()[1].equal(linear.bias) );

    // Test functionality
    torch::Tensor sample_input = torch::randn({32, 3});
    REQUIRE( linear(sample_input).sizes() ==  torch::IntArrayRef({32, 6}) );

    linear.weights = torch::tensor({1, 2, 3, 4, 5, 6, 7, 8, 9}).view({3, 3});
    linear.bias = torch::tensor({0, 0, 0});
    torch::Tensor test_input = torch::tensor({1, 2, 3, 4, 5, 6}).view({2, 3});
    torch::Tensor test_output = torch::tensor({14, 32, 50, 32, 77, 122}).view({2, 3});
    REQUIRE( linear(test_input).equal(test_output) );
}