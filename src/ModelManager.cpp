#include "models.h"
#include "layers.h"
#include <torch/torch.h>

class ModelManager
{
    public:
    ModelManager(std::shared_ptr<Module> model) : model(model) {}

    private:
    std::shared_ptr<Module> model;
};

int main()
{
    return 0;
}