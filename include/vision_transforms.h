#ifndef VISION_TRANSFORMS
#define VISION_TRANSFORMS

#include <memory>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <variant>
#include <vector>

using TransformResult = std::variant<cv::Mat, torch::Tensor>;

class Transform
{
  public:
    // Virtual method to apply the transformation
    virtual TransformResult apply(cv::Mat input) = 0;

    // perfect forwarding
    template <typename... Args> TransformResult operator()(Args &&...args);
};

// A class to compose multiple Transforms into one
class Compose : public Transform
{
  public:
    Compose(std::initializer_list<std::shared_ptr<Transform>> transforms);

    TransformResult apply(cv::Mat image) override;

  private:
    std::vector<std::shared_ptr<Transform>> transforms;
};

// Transforms cv::Mat to tensor
class ToTensor : public Transform
{
  public:
    TransformResult apply(cv::Mat image) override;
};

// Resizes cv::Mat to any shape
class Resize : public Transform
{
  public:
    Resize(int width, int height);
    TransformResult apply(cv::Mat input);

  private:
    int width;
    int height;
};

template <typename... Args> TransformResult Transform::operator()(Args &&...args)
{
    return apply(std::forward<Args>(args)...);
}

#endif