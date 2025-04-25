#include "fisheye_dewarper.h"
#include <chrono>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include "nlohmann/json.hpp"

#define CUDA_CHECK(call) \
{ cudaError_t err = call; if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            throw std::runtime_error("CUDA error"); \
    }}

// Use nlohmann/json namespace
using json = nlohmann::json;

FisheyeDewarper::FisheyeDewarper(int input_width, int input_height,
                                 int output_width, int output_height,
                                 dim3 block_size)
    : in_w_(input_width), in_h_(input_height),
    out_w_(output_width), out_h_(output_height),
    block_size_(block_size)
{
    if (in_w_ <= 0 || in_h_ <= 0 || out_w_ <= 0 || out_h_ <= 0) {
        throw std::invalid_argument("Invalid image dimensions");
    }

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    CUDA_CHECK(cudaMallocArray(&cu_array_, &channelDesc, in_w_, in_h_));
}

FisheyeDewarper::~FisheyeDewarper() {
    if (cu_array_) {
        cudaFreeArray(cu_array_);
        cu_array_ = nullptr;
    }
    if (tex_obj_) {
        cudaDestroyTextureObject(tex_obj_);
        tex_obj_ = 0;
    }
}

void FisheyeDewarper::process(const uchar4* d_input, uchar4* d_output,
                              const DewarpParams& params) {
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr((void*)d_input,
                                            in_w_ * sizeof(uchar4),
                                            in_w_, in_h_);
    copyParams.dstArray = cu_array_;
    copyParams.extent = make_cudaExtent(in_w_, in_h_, 1);
    copyParams.kind = cudaMemcpyDeviceToDevice;
    CUDA_CHECK(cudaMemcpy3D(&copyParams));

    // Debug: Copy cu_array_ back to host and inspect
    uchar4* h_debug_array = new uchar4[in_w_ * in_h_];
    cudaMemcpy3DParms debugCopyParams = {0};
    debugCopyParams.srcArray = cu_array_;
    debugCopyParams.dstPtr = make_cudaPitchedPtr(h_debug_array, in_w_ * sizeof(uchar4), in_w_, in_h_);
    debugCopyParams.extent = make_cudaExtent(in_w_, in_h_, 1);
    debugCopyParams.kind = cudaMemcpyDeviceToHost;
    CUDA_CHECK(cudaMemcpy3D(&debugCopyParams));

    int offset = (in_h_ / 4) * in_w_ + (in_w_ / 4);
    std::cerr << "cu_array_ pixel at (width/4, height/4): R=" << (int)h_debug_array[offset].x
              << " G=" << (int)h_debug_array[offset].y
              << " B=" << (int)h_debug_array[offset].z
              << " A=" << (int)h_debug_array[offset].w << std::endl;
    delete[] h_debug_array;

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cu_array_;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear; // Enable bilinear interpolation
    texDesc.readMode = cudaReadModeNormalizedFloat; // Read as normalized floats
    texDesc.normalizedCoords = true;

    if (tex_obj_) {
        cudaDestroyTextureObject(tex_obj_);
        tex_obj_ = 0;
    }
    CUDA_CHECK(cudaCreateTextureObject(&tex_obj_, &resDesc, &texDesc, nullptr));

    dim3 grid_size((out_w_ + block_size_.x - 1) / block_size_.x,
                   (out_h_ + block_size_.y - 1) / block_size_.y);

    auto start = std::chrono::high_resolution_clock::now();

    launchDewarpKernel(d_output, out_w_, out_h_,
                       params.hfov, params.vfov,
                       params.zoom,
                       params.pan, params.tilt, params.roll,
                       params.zoom_center_x, params.zoom_center_y,
                       (int)params.model,
                       tex_obj_, grid_size, block_size_);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration<float, std::milli>(end-start).count();

    metrics_.frame_time_ms = 0.9f * metrics_.frame_time_ms + 0.1f * ms;
    metrics_.throughput_mpixels = (out_w_ * out_h_) / (ms * 1000.0f);
    metrics_.total_frames++;
}

FisheyeDewarper::PerformanceMetrics FisheyeDewarper::getMetrics() const {
    return metrics_;
}

void FisheyeDewarper::saveConfig(const std::string& filename, const std::string& camera_name,
                                 const std::vector<ViewConfig>& views) const {
    json config;
    config["camera_name"] = camera_name;
    json viewsArray = json::array();
    for (const auto& view : views) {
        json viewObj;
        viewObj["name"] = view.name;
        viewObj["pan"] = view.pan;
        viewObj["tilt"] = view.tilt;
        viewObj["roll"] = view.roll;
        viewObj["zoom"] = view.zoom;
        viewObj["hfov"] = view.hfov;
        viewObj["vfov"] = view.vfov;
        viewObj["output_width"] = view.output_width;
        viewObj["output_height"] = view.output_height;
        viewObj["projection_model"] = static_cast<int>(view.model);
        viewsArray.push_back(viewObj);
    }
    config["views"] = viewsArray;

    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    file << config.dump(4); // Pretty print with 4-space indentation
    file.close();
}

void FisheyeDewarper::loadConfig(const std::string& filename, std::string& camera_name,
                                 std::vector<ViewConfig>& views) const {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }

    json config;
    file >> config;
    file.close();

    camera_name = config["camera_name"].get<std::string>();
    views.clear();
    for (const auto& viewJson : config["views"]) {
        ViewConfig view;
        view.name = viewJson["name"].get<std::string>();
        view.pan = viewJson["pan"].get<float>();
        view.tilt = viewJson["tilt"].get<float>();
        view.roll = viewJson["roll"].get<float>();
        view.zoom = viewJson["zoom"].get<float>();
        view.hfov = viewJson["hfov"].get<float>();
        view.vfov = viewJson["vfov"].get<float>();
        view.output_width = viewJson["output_width"].get<int>();
        view.output_height = viewJson["output_height"].get<int>();
        view.model = static_cast<ProjectionModel>(viewJson["projection_model"].get<int>());
        views.push_back(view);
    }
}
