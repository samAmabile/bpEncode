cmake_minimum_required(VERSION 3.12)
project(ling_data_tools)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(bpEncode bpe.cpp)
target_include_directories(bpEncode PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

