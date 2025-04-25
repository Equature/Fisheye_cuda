#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>

class FisheyeDewarper {
public:
    enum ProjectionModel {
        PROJECTION_EQUIDISTANT = 0,
        PROJECTION_EQUISOLID,
        PROJECTION_STEREOGRAPHIC,
        PROJECTION_ORTHOGRAPHIC,
        PROJECTION_EQUIRECTANGULAR,
        PROJECTION_PANINI,
        PROJECTION_CYLINDRICAL
    };

    struct DewarpParams {
        float hfov = 1.0f;
        float vfov = 1.0f;
        float pan = 0.0f;
        float tilt = 0.0f;
        float roll = 0.0f;
        float zoom = 1.0f;
        float zoom_center_x = 0.5f;
        float zoom_center_y = 0.5f;
        ProjectionModel model = PROJECTION_EQUISOLID;
        bool use_bilinear = true;
    };

    struct ViewConfig {
        std::string name;
        float pan;
        float tilt;
        float roll;
        float zoom;
        float hfov;
        float vfov;
        int output_width;
        int output_height;
        ProjectionModel model;
    };

    struct PerformanceMetrics {
        float frame_time_ms = 0;
        float throughput_mpixels = 0;
        size_t total_frames = 0;
    };

    FisheyeDewarper(int input_width, int input_height,
                    int output_width, int output_height,
                    dim3 block_size = {16,16});

    ~FisheyeDewarper();

    void process(const uchar4* d_input, uchar4* d_output, const DewarpParams& params);
    PerformanceMetrics getMetrics() const;

    // Configuration file handling
    void saveConfig(const std::string& filename, const std::string& camera_name,
                    const std::vector<ViewConfig>& views) const;
    void loadConfig(const std::string& filename, std::string& camera_name,
                    std::vector<ViewConfig>& views) const;

private:
    void setupTexture(const uchar4* d_input);
    void cleanupTexture();

    cudaTextureObject_t tex_obj_ = 0;
    cudaArray* cu_array_ = nullptr;
    int in_w_, in_h_;
    int out_w_, out_h_;
    dim3 block_size_;
    PerformanceMetrics metrics_;
};

extern "C" void launchDewarpKernel(
    uchar4* d_output, int width, int height,
    float hfov, float vfov, float zoom,
    float pan, float tilt, float roll,
    float zoom_center_x, float zoom_center_y,
    int projection_model,
    cudaTextureObject_t tex_obj,
    dim3 grid_size, dim3 block_size);
