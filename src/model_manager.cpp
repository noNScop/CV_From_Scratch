#include "layers.h"
#include "models.h"
#include <torch/torch.h>

class ModelManager
{
  public:
    ModelManager(std::shared_ptr<Module> model) : model(model)
    {
    }

  private:
    std::shared_ptr<Module> model;

    void train_step()
    {
        float avg_accuracy = 0;
        float avg_loss = 0;
        model->set_training(true);
    }
};