cmake_minimum_required(VERSION 3.18)
project(FisheyeDewarper LANGUAGES CXX CUDA)

# Set policies to suppress warnings
cmake_policy(SET CMP0146 NEW)
cmake_policy(SET CMP0104 NEW)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Qt setup
find_package(QT NAMES Qt6 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)

# Modern CUDA setup
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES "75")

# Configuration options
option(USE_MIPMAPS "Enable mipmap filtering" OFF)
option(USE_DYNAMIC_LOD "Enable dynamic level-of-detail" OFF)
option(ENABLE_PROFILING "Enable performance profiling" ON)
set(MAX_ZOOM 4.0 CACHE STRING "Maximum zoom factor")

# Source files
set(SOURCES
    src/fisheye_dewarper.cpp
    src/fisheye_dewarper.cu
    src/mainwindow.cpp
)

set(HEADERS
    include/fisheye_dewarper.h
    external/nlohmann/json.hpp
    include/mainwindow.h
)

set(QT_SOURCES
    src/main.cpp
)

# CUDA compilation options
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    -O3 \
    --use_fast_math \
    -Xcompiler=-fPIC"
)

# Create library
add_library(fisheye_dewarper STATIC
    ${SOURCES}
    ${HEADERS}
)
target_include_directories(fisheye_dewarper PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_libraries(fisheye_dewarper PUBLIC
    Qt${QT_VERSION_MAJOR}::Widgets
    CUDA::cudart_static
)
set_target_properties(fisheye_dewarper PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Qt application
add_executable(fisheye_dewarper_demo
    ${QT_SOURCES}
)
target_link_libraries(fisheye_dewarper_demo PRIVATE
    fisheye_dewarper
    Qt${QT_VERSION_MAJOR}::Widgets
    CUDA::cudart_static
)

# Installation
include(GNUInstallDirs)
install(TARGETS fisheye_dewarper
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Handle debug/release configurations
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configurations" FORCE)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()
