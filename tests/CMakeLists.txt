find_package(Catch2 3 REQUIRED)
# These tests can use the Catch2-provided main
add_executable(tests 
    test.cpp
    test_tensor.cpp
    ../src/layers.cpp
    ../src/vision_transforms.cpp
    ../src/data_utils.cpp
    ../src/models.cpp
    ../src/train.cpp
)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain ${OpenCV_LIBS})

# Set C++ standard
set_property(TARGET tests PROPERTY CXX_STANDARD 17)

include(CTest)
include(Catch)
catch_discover_tests(tests)