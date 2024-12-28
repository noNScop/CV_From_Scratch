cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(CV_From_Scratch)

# Define the base data directory relative to the CMakeLists.txt file
set(DATA_DIR "${CMAKE_SOURCE_DIR}/data")

# Pass the data directory as a preprocessor definition
add_definitions(-DDATA_DIR="${DATA_DIR}")

set(CMAKE_PREFIX_PATH
    "${CMAKE_SOURCE_DIR}/lib/Catch2"
    "${CMAKE_SOURCE_DIR}/lib/opencv/build"
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CEREAL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/lib/cereal/include")
include_directories(${CEREAL_INCLUDE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

find_package(OpenCV)

# Include directories for header files
include_directories(include ${OpenCV_INCLUDE_DIRS})

# Add the main executable with source files
add_executable(CV_From_Scratch 
    src/layers.cpp
    src/vision_transforms.cpp
    src/data_utils.cpp
    src/models.cpp
    src/train.cpp
    src/demo.cpp
)

# Link OpenCV
target_link_libraries(CV_From_Scratch ${OpenCV_LIBS})

# Add optimisation flag
target_compile_options(CV_From_Scratch PRIVATE -O3)

# Set C++ standard
set_property(TARGET CV_From_Scratch PROPERTY CXX_STANDARD 17)

# Add tests directory if tests are enabled
option(ENABLE_TESTS "Enable unit tests" ON)
if (ENABLE_TESTS)
    add_subdirectory(tests)
endif()