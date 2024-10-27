cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(CV_From_Scratch)

set(CMAKE_PREFIX_PATH 
"/home/vertex/Desktop/CV_From_Scratch/lib/libtorch"
"/home/vertex/Desktop/CV_From_Scratch/lib/Catch2"
)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Include directories for header files
include_directories(include)

# Add the main executable with source files
add_executable(CV_From_Scratch src/layers.cpp src/models.cpp src/train.cpp src/demo.cpp)

# Link Libtorch
target_link_libraries(CV_From_Scratch "${TORCH_LIBRARIES}")

# Set C++ standard
set_property(TARGET CV_From_Scratch PROPERTY CXX_STANDARD 17)

# Add tests directory if tests are enabled
option(ENABLE_TESTS "Enable unit tests" ON)
if (ENABLE_TESTS)
    add_subdirectory(tests)
endif()