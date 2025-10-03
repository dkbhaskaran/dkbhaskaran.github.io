---
title : Custom Memory Allocators in OpenCV - Zero copy for GPUs
date : 2025-08-19 00:00:00 +0800
categories: [Cuda, Zero-copy]
tags : [cuda, Zero-copy, unified-memory]
math : false
description : Zero copy for GPUs
toc : true
---


# Introduction

OpenCV provides hooks to control memory allocation. While small POD types like `cv::Point` and `cv::Rect` live on the stack, large arrays such as `cv::Mat` and `cv::cuda::GpuMat` allocate heap or device memory—and that path is customizable. On the CPU, a custom `cv::MatAllocator` can be installed; on CUDA, `cv::cuda::GpuMat::Allocator` serves the same role. (For OpenCL, `cv::UMat` follows a separate path.) With these mechanisms, developers can introduce pooling, alignment, NUMA-aware placement, pinned memory for faster DMA, and even **zero-copy** flows where the GPU accesses host RAM directly.


### CUDA zero-copy primitives: pinned/mapped host memory

Zero-copy means the GPU can directly access host RAM without `cudaMemcpy`. CUDA gives you two main ways to get there:

#### 1. `cudaHostAlloc`: allocate new pinned (optionally mapped) host memory

`cudaHostAlloc` returns pinned (page-locked) host memory. Its `flags` argument controls behavior; if you pass `cudaHostAllocMapped`, the allocation is also mapped into the GPU’s virtual address space so kernels can dereference it directly (*zero-copy*).

**Supported flags (bitwise-ORable):**
- `cudaHostAllocDefault` — Pinned memory only; no device mapping.
- `cudaHostAllocPortable` — Pinned allocation is visible to all CUDA contexts in the process.
- `cudaHostAllocMapped` — Pinned and mapped; obtain a device alias with `cudaHostGetDevicePointer`.
- `cudaHostAllocWriteCombined` — Write-combined pinned memory: fast CPU→GPU writes, slow CPU reads (best for upload buffers).

**Prerequisite (for mapping):** enable host mapping before creating the CUDA context.
```cpp
cudaSetDevice(0);
cudaSetDeviceFlags(cudaDeviceMapHost);  // enable host mapping
cudaFree(0);                            // create context with these flags
```

**Freeing:** release with `cudaFreeHost(ptr)` (do not use `free`/`delete`, and do not call `cudaHostUnregister` for `cudaHostAlloc` allocations).

#### 2. `cudaHostRegister`: pin/map an existing host buffer (retrofit zero-copy)

Use `cudaHostRegister(ptr, size, flags)` when a buffer already exists and you want pinned/zero-copy without reallocating.  
It pins (page-locks) the range for fast DMA; with `cudaHostRegisterMapped` it also maps those pages into the GPU’s virtual address space so kernels can dereference them directly.

**Flags:**
- `cudaHostRegisterDefault` — pin only.
- `cudaHostRegisterMapped` — pin **+** map (enables zero-copy).
- `cudaHostRegisterPortable` — usable from any CUDA context in the process.
- `cudaHostRegisterReadOnly` — hint that the GPU will only read.

**Prerequisite for mapping:** enable host mapping before the CUDA context is created. A simple usage would be like this

```cpp
void*  buf   = existingCpuPtr;                         // e.g., img.data from cv::Mat
size_t bytes = totalSizeInBytes;

cudaHostRegister(buf, bytes, cudaHostRegisterMapped);  // pin + map
void* d = nullptr;
cudaHostGetDevicePointer(&d, buf, 0);                  // device alias (zero-copy)
// ... launch kernels using 'd' ...
cudaHostUnregister(buf);                               // then free/delete 'buf'
```

### An example program using cudaHostAlloc

This program demonstrates a zero-copy OpenCV + CUDA pipeline:
- **Enable host mapping:** Initializes CUDA with `cudaSetDeviceFlags(cudaDeviceMapHost)` and checks `canMapHostMemory`.
- **Custom `cv::Mat` allocator:** Installs a `MatAllocator` that uses `cudaHostAlloc(..., cudaHostAllocMapped)` so `cv::Mat` buffers live in **pinned + mapped** host RAM (DMA-friendly and visible to the GPU).
- **Header-only GPU view:** Wraps those mapped `cv::Mat` buffers as `cv::cuda::GpuMat` (via a helper like `wrappedGpuMat` using `cudaHostGetDevicePointer`)—**no `cudaMemcpy`**.
- **CUDA image ops:** Applies a Gaussian blur (`cv::cuda::createGaussianFilter()->apply`) followed by Canny edge detection (`cv::cuda::createCann

```cpp
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp> // cv::cuda::createGaussianFilter (alternative)
#include <opencv2/cudaimgproc.hpp> // cv::cuda::GaussianBlur, CannyEdgeDetector
#include <opencv2/imgcodecs.hpp>

using namespace cv;

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(e));                                          \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

// A custom allocator for cv::Mat. This will allocate a pinned and
// mapped host memory that can be used directly by GPU.
class GpuMappedAllocator : public MatAllocator {
public:
  UMatData *allocate(int dims, const int *sizes, int type, void *data,
                     size_t *step, AccessFlag flags,
                     UMatUsageFlags usageFlags) const override {
    // Working with 2 dimensions for now.
    const int rows = sizes[0], cols = sizes[1];
    const size_t elementSize = CV_ELEM_SIZE(type);
    const size_t rowStep = cols * elementSize;

    if (step) {
      // OpenCV expects step[0] = row stride in bytes
      // step[1] = elementSize
      step[0] = rowStep;
      if (dims > 1)
        step[1] = CV_ELEM_SIZE(type);
    }

    UMatData *umat = new UMatData(this);
    umat->size = rows * rowStep;

    if (data) {
      // Wrap user memory (not owning)
      umat->data = umat->origdata = static_cast<uchar *>(data);
      umat->flags = UMatData::USER_ALLOCATED;
      umat->handle = nullptr;
      return umat;
    }

    void *host = nullptr;
    CUDA_CHECK(cudaHostAlloc(&host, rows * rowStep, cudaHostAllocMapped));
    umat->data = umat->origdata = static_cast<uchar *>(host);
    umat->handle = host;
    return umat;
  }

  virtual bool allocate(UMatData *data, AccessFlag accessflags,
                        UMatUsageFlags usageFlags) const override {
    // Not supporting UMat/OpenCL path; nothing to do.
    return false;
  }
  class GpuMappedAllocator : public MatAllocator {
public:
  UMatData *allocate(int dims, const int *sizes, int type, void *data,
                     size_t *step, AccessFlag flags,
                     UMatUsageFlags usageFlags) const override {
    // Working with 2 dimensions for now.
    const int rows = sizes[0], cols = sizes[1];
    const size_t elementSize = CV_ELEM_SIZE(type);
    const size_t rowStep = cols * elementSize;

    if (step) {
      // OpenCV expects step[0] = row stride in bytes
      // step[1] = elementSize
      step[0] = rowStep;
      if (dims > 1)
        step[1] = CV_ELEM_SIZE(type);
    }

    UMatData *umat = new UMatData(this);
    umat->size = rows * rowStep;

    if (data) {
      // Wrap user memory (not owning)
      umat->data = umat->origdata = static_cast<uchar *>(data);
      umat->flags = UMatData::USER_ALLOCATED;
      umat->handle = nullptr;
      return umat;
    }

    void *host = nullptr;
    CUDA_CHECK(cudaHostAlloc(&host, rows * rowStep, cudaHostAllocMapped));
    umat->data = umat->origdata = static_cast<uchar *>(host);
    umat->handle = host;
    return umat;
  }

  virtual bool allocate(UMatData *data, AccessFlag accessflags,
                        UMatUsageFlags usageFlags) const override {
    // Not supporting UMat/OpenCL path; nothing to do.
    return false;
  }

int main() {
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
  CUDA_CHECK(cudaFree(0)); // force context

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  if (!prop.canMapHostMemory) {
    fprintf(stderr, "Device cannot map host memory.\n");
    return 1;
  }

  // Install the custom Mat allocator (affects *Mat* only)
  static GpuMappedAllocator gpuAllocator;
  Mat::setDefaultAllocator(&gpuAllocator);

  // Create a custom input matrix
  const int W = 640, H = 480;
  Mat in(H, W, CV_8UC1);
  Mat out(H, W, CV_8UC1);

  for (int y = 0; y < H; y++) {
    auto *row = in.ptr<uchar>(y);
    for (int x = 0; x < W; x++) {
      row[x] = static_cast<uchar>((x / 8) % 2 ? 220 : 35);
    }
  }

  cuda::GpuMat gin = wrappedGpuMat(in);
  cuda::GpuMat gout = wrappedGpuMat(out);

  cuda::GpuMat tmp;

  cv::Ptr<cv::cuda::Filter> gauss = cv::cuda::createGaussianFilter(
      gin.type(), gin.type(), cv::Size(5, 5), 1.2, 1.2, cv::BORDER_DEFAULT);
  cv::Ptr<cv::cuda::CannyEdgeDetector> canny =
      cv::cuda::createCannyEdgeDetector(50, 150);

  gauss->apply(gin, tmp);
  canny->detect(tmp, gout);

  imwrite("input.png", in);
  imwrite("output.png", out);

  return 0;
}
```

#### Appendix : Downloading and building OpenCV for Nvidia GPUs
##### Building OpenCV from source

These steps will download build and install openCV in your local folder with cuda support. Note cuda toolkit and cuDNN must be installed before this

```cmake
$ mkdir -p ~/src && cd ~/src
$ git clone --depth 1 https://github.com/opencv/opencv.git
$ git clone --depth 1 https://github.com/opencv/opencv_contrib.git
$ mkdir -p opencv-build && cd opencv-build

$ cmake ../opencv \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/.local/opencv-cuda \
  -DWITH_CUDA=ON -DOPENCV_DNN_CUDA=ON -DWITH_CUDNN=ON -DWITH_CUBLAS=ON \
  -DENABLE_FAST_MATH=ON -DCUDA_FAST_MATH=ON -DCUDA_ARCH_BIN=8.9 \
  -DBUILD_opencv_cudacodec=ON -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
  -DBUILD_EXAMPLES=OFF

$ make -j 16
$ make install  # It will install in ~/.local/opencv-cuda

$ export OpenCV_DIR=$HOME/.local/opencv-cuda/lib/cmake/opencv4
$ export LD_LIBRARY_PATH=$HOME/.local/opencv-cuda/lib:$LD_LIBRARY_PATH
```

Once installed use this program to check 
```cpp
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

int main() {
  int n = cv::cuda::getCudaEnabledDeviceCount();
  std::cout << "CUDA devices: " << n << "\n";
  if (n > 0)
    cv::cuda::printShortCudaDeviceInfo(0);
}
```

Build the above program using 
```make
export OPENCV_DIR=$HOME/.local/opencv-cuda/
g++ -std=c++17 check_cuda.cpp -o check_cuda $(pkg-config --cflags --libs opencv4)

# The output looks like the below in my case 
CUDA devices: 1
Device 0:  "NVIDIA GeForce RTX 4050 Laptop GPU"  6140Mb, sm_89, Driver/Runtime ver.12.70/12.60
```
