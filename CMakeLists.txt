cmake_minimum_required(VERSION 3.15)
project(bpEncode)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

add_library(bpe_lib src/bpEncode/bpe.cpp)
set_target_properties(bpe_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(bpe_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(_bpEncode MODULE src/bpEncode/binding.cpp)

#set_target_properties(_bpEncode PROPERTIES
#    PREFIX ""
#    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/bpEncode"
#)

set_target_properties(_bpEncode PROPERTIES PREFIX "")

target_link_libraries(_bpEncode PRIVATE bpe_lib pybind11::module)

#set_target_properties(_bpEncode PROPERTIES PREFIX "")
#set_target_properties(bpEncode PROPERTIES OUTPUT_NAME "_bpEncode")

install(TARGETS _bpEncode DESTINATION bpEncode)


