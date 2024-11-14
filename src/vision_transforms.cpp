#include "vision_transforms.h"

using TransformResult = std::variant<cv::Mat, torch::Tensor>;

Compose::Compose(std::initializer_list<std::shared_ptr<Transform>> transforms) : transforms(transforms)
{
}

// Apply internaly stored transforms sequentialy
TransformResult Compose::apply(cv::Mat image)
{
    TransformResult output = image;
    for (auto const &transform : transforms)
    {
        if (std::holds_alternative<cv::Mat>(output))
        {
            output = transform->apply(std::get<cv::Mat>(output));
        }
        else if (std::holds_alternative<torch::Tensor>(output))
        {
            throw std::runtime_error("Image is already a Tensor, cannot apply further transforms");
        }
    }
    return output;
}

// Deliberately not using high level function (which is probably faster) for transforming cv::Mat into torch::Tensor
// as later torch::Tensors are supposed to be substituted with custom Tensor
TransformResult ToTensor::apply(cv::Mat image)
{
    int channels = image.channels();
    int height = image.rows;
    int width = image.cols;

    torch::Tensor tensor = torch::zeros({channels, height, width}, torch::kFloat32);

    for (int c = 0; c < channels; ++c)
    {
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                tensor[c][h][w] = image.at<cv::Vec3b>(h, w)[c] / 255.0;
            }
        }
    }

    return tensor;
}

Resize::Resize(int width, int height) : width(width), height(height)
{
}

TransformResult Resize::apply(cv::Mat input)
{
    cv::resize(input, input, cv::Size(width, height));
    return input;
}