# CV_From_Scratch

**CV_From_Scratch** is a project where we implemented a **PyTorch-like library in C++** for **Computer Vision** tasks. This library includes layers, a CNN architecture, a `Learner` class, data loaders, and basic image transforms and `Tensor` class, all from scratch. The project uses **OpenCV** for image processing and **Cereal** for serialisation.

## Features
- Custom Tensor
- Implemented CNN architecture and layers from scratch in C++.
- DataLoader and Datasets with basic transformations using OpenCV.
- A simple demonstration app for training and testing a CNN model.
- Unit tests for verification (optional).

## Usage

While this project doesn't support **CUDA** and isn't designed for high-performance usage, it includes a basic **terminal-based demonstration app**. This app allows you to:
- Train a CNN.
- Run inference on the trained model.
- Save and load model parameters to a file.

To use the app, compile the project and run the **`CV_From_Scratch`** executable.

There are also **unit tests** located in the `tests/` directory that can be executed by running `tests/tests` after compilation.

### Setup

To set up the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd CV_From_Scratch

2. **Download the MNIST dataset**: Run the following Python script to download the MNIST dataset and place the images in the `data/` directory: `python get_data.py`

3. **Download required libraries**: You need to download the following libraries and place them in the `lib/` directory:
- [Cereal](https://uscilab.github.io/cereal/index.html)
- [OpenCV](https://github.com/opencv/opencv/tree/4.10.0)
- [Catch2 (for tests)](https://github.com/catchorg/Catch2?tab=readme-ov-file)

  The details regarding setting up all the 3 above mentioned libraries is in the `README.md` file inside `lib/`

**Note**: If you do not want to use tests, you can skip Catch2 by disabling tests in the CMakeLists.txt file:
```CMake
# Add tests directory if tests are enabled
option(ENABLE_TESTS "Enable unit tests" ON) # Set ON -> OFF to disable tests
if (ENABLE_TESTS)
    add_subdirectory(tests)
endif()
```
4. **Build and compile the project**: Follow the standard CMake build process:
```bash
mkdir build
cd build
cmake ..
make
```
5. **Run the demonstration app**: After compilation, run the executable:
`./CV_From_Scratch`

## Project Structure
Here’s a quick overview of the project structure:
```bash
CV_From_Scratch/
├── build/             # A place for compiling the project
├── data/              # MNIST dataset
├── include/           # Header files for CNN, layers, data loading, etc.
├── lib/               # External libraries (LibTorch, OpenCV, Catch2)
├── src/               # Source code for CNN, layers, data loading, etc.
├── tests/             # Unit tests (optional)
├── CMakeLists.txt     # CMake configuration
├── get_data.py        # Download MNIST dataset
└── README.md          # Project description
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.