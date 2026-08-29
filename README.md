# GPU IPC

The first fully GPU-optimized IPC framework and source code for
**[GIPC: Fast and Stable Gauss-Newton Optimization of IPC Barrier Energy](https://dl.acm.org/doi/10.1145/3643028)**,
*ACM Transactions on Graphics*, 2024.

GPU IPC is also useful as a benchmark for research on GPU collision handling,
barrier methods, and nonlinear finite-element simulation.

**Authors:** Kemeng Huang, Floyd M. Chitalu, Huancheng Lin, Taku Komura
**Source code contributor:** [Kemeng Huang](https://kemenghuang.github.io)

[![Watch the video](https://github.com/KemengHuang/GPU_IPC/blob/main/Assets/video1.png)](https://youtu.be/5rwp6AiHtrw)

## Requirements

- NVIDIA CUDA-capable GPU
- Windows or Linux
- CMake 3.24 or newer
- C++17 compiler and a compatible CUDA Toolkit
- Eigen 3.4, freeglut 3.4, and GLEW 2.2

## Build

By default CMake builds device code for the GPU installed on the build host.
For a distributable binary, set `CMAKE_CUDA_ARCHITECTURES` explicitly (for
example, `86` for Ampere, `89` for Ada, or `120` for Blackwell).

### Linux

```bash
sudo apt install libglew-dev freeglut3-dev libeigen3-dev

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Windows

The dependencies can be installed with [vcpkg](https://github.com/microsoft/vcpkg).
Set `CMAKE_TOOLCHAIN_FILE` to
`<vcpkg>/scripts/buildsystems/vcpkg.cmake`, then run:

```bat
vcpkg install eigen3 freeglut glew

cmake -S . -B build
cmake --build build --config Release
```

To override the detected GPU architecture:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
```

## Run

```bash
# Linux
./build/gipc

# Windows
build\Release\gipc.exe
```

A GLUT window opens the default simulation scene. Asset and output paths are
compiled through `GIPC_ASSETS_DIR` and `GIPC_OUTPUT_DIR`, so the executable can
be launched from any working directory.

### Viewer controls

| Input | Action |
| --- | --- |
| mouse drag | rotate view |
| `w` `s` `a` `d` `q` `e` | translate view |
| `space` | pause or resume simulation |
| `k` | toggle surface rendering |
| `f` | toggle BVH rendering |
| `9` | toggle surface export |
| `/` | toggle screenshot capture |

## Configuration

Simulation parameters live in `Assets/scene/parameterSetting.txt`, including
material properties, time step, solver tolerances, friction, and the relative
collision distance threshold. Restart the executable after editing the file;
recompilation is not required.

### Dynamic GPU memory

Collision and Hessian storage grows automatically:

- DCD pairs start at 100,000 entries.
- CCD pairs start at 1,000,000 entries.
- A collision pass that exceeds capacity finishes counting without writing out
  of bounds, grows geometrically, and reruns to regenerate the complete list.
- Hessian block arrays are resized from the current fixed-energy and contact
  counts before assembly.
- MAS workspaces and friction/close-constraint history retain capacity across
  frames.
- CUB scan, radix-sort, and reduction operations share a persistent temporary
  workspace, so steady-state frames do not allocate CUB scratch memory.

The old `collision_detection_buff_scale` setting is retained for compatibility
and diagnostics, but correctness no longer depends on manually tuning a maximum
pair count.

## Batch and memory validation

Set `GIPC_STEPS` to run without entering the viewer loop. `GIPC_STEPS=0` loads
and initializes the scene only. `GIPC_DUMP_STATE` writes packed final `double3`
vertex positions. Set `GIPC_WRITE_TIMING=1` only when the per-frame
`timeCost.txt` report is needed; normal runs avoid that disk write.

```bash
GIPC_STEPS=10 GIPC_DUMP_STATE=state.bin ./build/gipc
```

`GIPC_STRESS_DYNAMIC_MEMORY=1` starts pair buffers at one entry and Hessian
buffers at zero. It is intended to exercise every dynamic-growth path:

```bash
GIPC_STEPS=1 GIPC_STRESS_DYNAMIC_MEMORY=1 \
  compute-sanitizer --tool memcheck --leak-check full ./build/gipc
```

## Project structure

```text
GPU_IPC/
├── GIPC.cu/.cuh             core IPC solver, CCD, line search, and Newton loop
├── PCG_SOLVER.cu/.cuh       block-Jacobi PCG and MAS-PCG
├── MASPreconditioner.*      multilevel additive Schwarz preconditioner
├── mlbvh.cu/.cuh            LBVH broad-phase collision detection
├── device_buffer.h          move-only persistent CUDA buffer ownership
├── cuda_cub_wrappers.*      persistent CUB scan/sort/reduction workspace
├── ACCD.cu/.cuh             continuous collision detection
├── femEnergy.cu/.cuh        FEM elasticity energy
├── gpu_eigen_libs.cuh       fixed-size GPU matrix operations
├── GIPC_PDerivative.cuh     analytic energy derivatives
├── gl_main.cpp              GLUT viewer and batch regression entry point
└── load_mesh.*              mesh I/O
```

## Citation

```bibtex
@article{gipc2024,
  author = {Huang, Kemeng and Chitalu, Floyd M. and Lin, Huancheng and Komura, Taku},
  title = {GIPC: Fast and Stable Gauss-Newton Optimization of IPC Barrier Energy},
  year = {2024},
  publisher = {Association for Computing Machinery},
  volume = {43},
  number = {2},
  doi = {10.1145/3643028},
  journal = {ACM Trans. Graph.},
  month = {mar},
  articleno = {23},
  numpages = {18}
}
```
