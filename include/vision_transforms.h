#ifndef VISION_TRANSFORMS
#define VISION_TRANSFORMS

#include <memory>
#include <opencv2/opencv.hpp>
#include "tensor.h"
#include <variant>
#include <vector>

using TransformResult = std::variant<cv::Mat, Tensor<float>>;

class Transform
{
  public:
    virtual TransformResult apply(cv::Mat input) = 0;

    template <typename... Args> TransformResult operator()(Args &&...args);
};

class Compose : public Transform
{
  public:
    Compose(std::initializer_list<std::shared_ptr<Transform>> transforms);

    TransformResult apply(cv::Mat image) override;

  private:
    std::vector<std::shared_ptr<Transform>> transforms;
};

class ToTensor : public Transform
{
  public:
    TransformResult apply(cv::Mat image) override;
};

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