#include <torch/torch.h>
#include "layers.h"
#include <iostream>

// ModelManager

int main()
{   
    Linear linear(10, 10);
    std::cout << *linear.parameters()[0] << std::endl;
}