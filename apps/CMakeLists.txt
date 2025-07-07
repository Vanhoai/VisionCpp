cmake_minimum_required(VERSION 3.18)
project(apps VERSION 1.0.0)

add_subdirectory(benchmark)
add_subdirectory(training)
add_subdirectory(study)

add_custom_target(${PROJECT_NAME})
add_dependencies(${PROJECT_NAME} benchmark training study)

# Print information about available applications
message(STATUS "+-------------------- VISION CPP APPLICATIONS ---------------------+")
message(STATUS "| 1: Training: Training models (from scratch) with custom datasets")
message(STATUS "| 2: Benchmark: Benchmark models with CPU|GPU computing")
message(STATUS "| 3. Study: Run anything you want :D")
message(STATUS "+------------------------------------------------------------------+")
