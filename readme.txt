fisheye_dewarper/
├── CMakeLists.txt
├── include/
│   ├── fisheye_dewarper.h
│   └── main_window.h
├── src/
│   ├── fisheye_dewarper.cpp
│   ├── fisheye_dewarper.cu
│   ├── main_window.cpp
│   └── main.cpp
└── external/
    └── nlohmann/include/json.hpp  # json library


This project builds a library (fisheye_dewarper.a) from 3 files: fisheye_dewarper.cpp/.h/.cu
The other files are used to create a Qt app that allows you to exercise the library.



Key Features of This Setup:

    Dual Compilation:

        CUDA code (.cu) compiled with NVCC

        C++/Qt code (.cpp) compiled with host compiler

    Qt Integration:

        Live parameter adjustment via sliders

        Image display using QLabel or QGraphicsView

        Async processing to keep UI responsive

    Configuration Options:

        Toggle mipmaps/LOD at compile-time

        Adjust max zoom factor

        Enable/disable profiling

    GPU Architecture:

        Uses native for best performance on your hardware

        Can specify manually (e.g., "75" for Turing, "86" for Ampere)

For Qt Creator integration:

    Open the CMake project directly

    Set up kit with CUDA support

    Add debug/release build configurations

    Set working directory to contain test images
