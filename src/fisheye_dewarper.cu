#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <vector_types.h>
#include <cstdio>

#define USE_MIPMAPS 0
#define USE_DYNAMIC_LOD 0
#define ENABLE_PROFILING 1
#define MAX_ZOOM 4.0f

cudaTextureObject_t fisheyeTex;

__device__ float3 rotatePoint(float3 p, float pan, float tilt, float roll) {
    float cp, sp, ct, st, cr, sr;
    sincosf(pan, &sp, &cp);
    sincosf(tilt, &st, &ct);
    sincosf(roll, &sr, &cr);

    const float x1 = cp * p.x - sp * p.z;
    const float z1 = sp * p.x + cp * p.z;
    const float y2 = ct * p.y - st * z1;
    const float z2 = st * p.y + ct * z1;

    return make_float3(
        cr * x1 - sr * y2,
        sr * x1 + cr * y2,
        z2
        );
}

__global__ void dewarpFisheyeKernel(
    uchar4* output,
    int width, int height,
    float hfov, float vfov,
    float zoom,
    float pan, float tilt, float roll,
    float zoom_center_x, float zoom_center_y,
    int projection_model,
    cudaTextureObject_t tex_obj,
    float* debug_values = nullptr)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float u = (x + 0.5f) / width;
    float v = (y + 0.5f) / height;

    // Map output coordinates to rectilinear projection
    float x_norm = (2.0f * (u - zoom_center_x)) * tanf(hfov / 2.0f) * zoom;
    float y_norm = (2.0f * (v - zoom_center_y)) * tanf(vfov / 2.0f) * zoom;

    // Create a 3D ray for rectilinear projection (camera looking along +z)
    float3 ray = make_float3(x_norm, -y_norm, 1.0f);

    // Apply pan, tilt, and roll rotations
    ray = rotatePoint(ray, -pan, -tilt, roll);

    // Normalize the ray
    float r = rsqrtf(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
    ray.x *= r; ray.y *= r; ray.z *= r;

    // Project onto a fisheye sphere with selected projection model
    float theta = atan2f(ray.y, ray.x);
    float phi = acosf(ray.z);
    float u_fish, v_fish;
    float r_fish = 0.0f; // Declare r_fish with a default value

    if (projection_model == 6) { // Cylindrical projection
        // Cylindrical projection: u_fish is proportional to theta, v_fish to z or phi
        u_fish = (theta + M_PI) / (2.0f * M_PI); // Map theta from [-π, π] to [0, 1]
        v_fish = 0.5f + 0.5f * (ray.z); // Map z from [-1, 1] to [0, 1]
        v_fish = 1.0f - v_fish; // Flip vertically to correct orientation
    } else {
        switch (projection_model) {
        case 0: // Equidistant
            r_fish = phi;
            break;
        case 1: // Equisolid
            r_fish = 2.0f * sinf(phi / 2.0f);
            break;
        case 2: // Stereographic
            r_fish = tanf(phi / 2.0f);
            break;
        case 3: // Orthographic
            r_fish = sinf(phi);
            break;
        case 4: // Equirectangular
            r_fish = phi / M_PI;
            break;
        case 5: // Panini
        {
            float d = 1.0f;
            float lambda = (d + 1.0f) / (d + cosf(phi));
            r_fish = lambda * sinf(phi);
            break;
        }
        default:
            r_fish = 2.0f * sinf(phi / 2.0f); // Default to equisolid
        }

        u_fish = 0.5f + 0.5f * r_fish * cosf(theta);
        v_fish = 0.5f + 0.5f * r_fish * sinf(theta);

        // Fix orientation: only apply vertical flip to match fisheye image orientation
        v_fish = 1.0f - v_fish; // Vertical flip (keep top of fisheye at top of dewarped image)
    }

    // Debug: Log intermediate values at (width/4, height/4)
    if (debug_values && x == width/4 && y == height/4) {
        debug_values[0] = u;
        debug_values[1] = v;
        debug_values[2] = x_norm;
        debug_values[3] = y_norm;
        debug_values[4] = ray.x;
        debug_values[5] = ray.y;
        debug_values[6] = ray.z;
        debug_values[7] = theta;
        debug_values[8] = phi;
        debug_values[9] = r_fish;
        debug_values[10] = u_fish;
        debug_values[11] = v_fish;
    }

    // Debug: Log texture coordinates at additional points
    if ((x == 0 && y == 0) || (x == width-1 && y == 0) || (x == 0 && y == height-1) || (x == width-1 && y == height-1)) {
        printf("Texture coords at (%d, %d): u_fish=%f, v_fish=%f\n", x, y, u_fish, v_fish);
    }

    uchar4 pixel;
    if (u_fish >= 0.0f && u_fish <= 1.0f && v_fish >= 0.0f && v_fish <= 1.0f) {
        // Texture is now read as normalized floats (float4 in [0, 1])
        float4 sampled = tex2D<float4>(tex_obj, u_fish, v_fish);
        // Convert back to uchar4 by scaling to [0, 255]
        pixel = make_uchar4(
            static_cast<unsigned char>(sampled.x * 255.0f),
            static_cast<unsigned char>(sampled.y * 255.0f),
            static_cast<unsigned char>(sampled.z * 255.0f),
            static_cast<unsigned char>(sampled.w * 255.0f)
            );
        if (x == width/4 && y == height/4) {
            printf("tex2D called at (width/4, height/4): u_fish=%f, v_fish=%f, pixel=(%d, %d, %d, %d)\n",
                   u_fish, v_fish, pixel.x, pixel.y, pixel.z, pixel.w);
        }
    } else {
        pixel = make_uchar4(0, 0, 255, 255); // Blue for out-of-bounds
        if (x == width/4 && y == height/4) {
            printf("Out-of-bounds at (width/4, height/4): u_fish=%f, v_fish=%f\n", u_fish, v_fish);
        }
    }

    output[y * width + x] = pixel;
}

extern "C" void launchDewarpKernel(
    uchar4* d_output, int width, int height,
    float hfov, float vfov, float zoom,
    float pan, float tilt, float roll,
    float zoom_center_x, float zoom_center_y,
    int projection_model,
    cudaTextureObject_t tex_obj,
    dim3 grid_size, dim3 block_size)
{
    float* d_debug_values;
    cudaMalloc(&d_debug_values, 12 * sizeof(float));
    cudaMemset(d_debug_values, 0, 12 * sizeof(float));

    dewarpFisheyeKernel<<<grid_size, block_size>>>(
        d_output, width, height,
        hfov, vfov, zoom,
        pan, tilt, roll,
        zoom_center_x, zoom_center_y,
        projection_model,
        tex_obj,
        d_debug_values);

    float h_debug_values[12];
    cudaMemcpy(h_debug_values, d_debug_values, 12 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Debug at (width/4, height/4):\n");
    printf("  u=%f, v=%f\n", h_debug_values[0], h_debug_values[1]);
    printf("  x_norm=%f, y_norm=%f\n", h_debug_values[2], h_debug_values[3]);
    printf("  ray=(%f, %f, %f)\n", h_debug_values[4], h_debug_values[5], h_debug_values[6]);
    printf("  theta=%f, phi=%f, r_fish=%f\n", h_debug_values[7], h_debug_values[8], h_debug_values[9]);
    printf("  u_fish=%f, v_fish=%f\n", h_debug_values[10], h_debug_values[11]);
    fflush(stdout);

    cudaFree(d_debug_values);
}
