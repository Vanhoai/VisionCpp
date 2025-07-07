cmake_minimum_required(VERSION 3.30)
project(nntest VERSION 1.0.0 LANGUAGES C CXX)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    benchmark::benchmark
)
