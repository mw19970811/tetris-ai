cmake_minimum_required(VERSION 3.20)

# Emscripten toolchain required for WASM builds.
# Build: emcmake cmake -B build && cmake --build build
if(NOT EMSCRIPTEN)
    message(FATAL_ERROR "Emscripten toolchain not detected. Use emcmake to configure.")
endif()

add_executable(tetris_ai_web
    tetris_ai_web.cpp
    ../model_loader.cpp
    ../player.cpp
)

target_include_directories(tetris_ai_web PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_SOURCE_DIR}/env/core
)

target_link_libraries(tetris_ai_web PRIVATE
    onnxruntime::onnxruntime
)

# Emscripten-specific settings.
set_target_properties(tetris_ai_web PROPERTIES
    LINK_FLAGS "-s WASM=1 -s ALLOW_MEMORY_GROWTH=1 -s MODULARIZE=1 -s EXPORT_NAME='TetrisAI'"
)
