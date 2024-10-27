find_package(Catch2 3 REQUIRED)
# These tests can use the Catch2-provided main
add_executable(tests 
    test.cpp
    ../src/layers.cpp
    ../src/models.cpp
    ../src/train.cpp
)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain "${TORCH_LIBRARIES}")

include(CTest)
include(Catch)
catch_discover_tests(tests)