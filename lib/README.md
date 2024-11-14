# Libraries Setup

## Versions
I am using the following versions of these libraries, although using a little older or newer versions should make much of a difference:
- **LibTorch** - **2.5.0+cpu**
- **OpenCV** - **4.10.0**
- **Catch2** - **3.7.1**

## LibTorch
Go to [LibTorch](https://pytorch.org/) and download LibTorch for your operating system from there, place it inside `lib/` directory with name `libtorch/`

## OpenCV
To set up **OpenCV**, follow these steps:

1. Go to the [OpenCV releases page](https://opencv.org/releases/) to download the source code.
2. Alternatively, you can clone the OpenCV repository from GitHub:
```bash
git clone <URL to OpenCV GitHub>
cd opencv
mkdir build
cd build
cmake ..
make
```
3. Ensure that the `opencv/` folder is placed inside the `lib/` directory, and that the `build/` folder is inside `opencv/`. If you choose a different directory structure, you may need to modify the `CMakeLists.txt` file accordingly.

## Catch2 (for tests)
If you want to use Catch2 for testing, follow these steps to set it up:
1. Clone the Catch2 repository from GitHub:
```bash
git clone https://github.com/catchorg/Catch2.git
cd Catch2
```
2. Build and install Catch2 with the following commands:
```bash
cmake -B build -S . -DBUILD_TESTING=OFF
sudo cmake --build build/ --target install
```
3. Once installed, you can find Catch2 in the `lib/` directory. However, Catch2 is optional if you disable tests in the `CMakeLists.txt` file.