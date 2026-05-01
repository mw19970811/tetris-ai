cmake_minimum_required(VERSION 3.20)

find_package(pybind11 REQUIRED)

pybind11_add_module(tetris_core
    tetris_bindings.cpp
)

target_link_libraries(tetris_core PRIVATE tetris_core_lib)

target_include_directories(tetris_core PRIVATE
    ${CMAKE_SOURCE_DIR}/env/core
)

install(TARGETS tetris_core DESTINATION lib)
