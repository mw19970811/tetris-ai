cmake_minimum_required(VERSION 3.20)
project(tetris_ai LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)   # 继续用旧的 FindPythonInterp/Libs
endif()
if(POLICY CMP0002)
  cmake_policy(SET CMP0002 NEW)   # 允许同名目标
endif()

# C++ Tetris environment core + pybind11 bindings.
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    add_subdirectory(env/core)
    add_subdirectory(env/bindings)
else()
    message(STATUS "pybind11 not found — C++ env acceleration disabled. "
                   "Install with: pip install pybind11")
endif()

# Inference module (optional — requires ONNX Runtime).
find_package(onnxruntime QUIET)
if(onnxruntime_FOUND)
    add_subdirectory(inference/cpp)
else()
    message(STATUS "ONNX Runtime not found — skipping inference/cpp. "
                   "Set onnxruntime_DIR to enable C++ inference.")
endif()
