cmake_minimum_required(VERSION 3.20)

find_package(onnxruntime REQUIRED)

add_library(tetris_inference STATIC
    model_loader.cpp
    player.cpp
)

target_include_directories(tetris_inference PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/env/core
)

target_link_libraries(tetris_inference PUBLIC
    onnxruntime::onnxruntime
)

# WebAssembly deployment (optional).
option(BUILD_WASM "Build WebAssembly inference module" OFF)
if(BUILD_WASM)
    add_subdirectory(web)
endif()
