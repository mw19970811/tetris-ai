cmake_minimum_required(VERSION 3.20)

add_library(tetris_core_lib STATIC
    piece_data.cpp
    randomizer.cpp
    tetris_core.cpp
    tetris_env.cpp
    action_gen.cpp
    state_encoder.cpp
)

target_include_directories(tetris_core_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_features(tetris_core_lib PUBLIC cxx_std_17)

if(MSVC)
    target_compile_options(tetris_core_lib PRIVATE /O2 /arch:AVX2)
else()
    target_compile_options(tetris_core_lib PRIVATE -O3 -march=native -flto)
endif()
