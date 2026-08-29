# GPU IPC

The first fully GPU-optimized IPC framework — source code of the paper
**[GIPC: Fast and Stable Gauss-Newton Optimization of IPC Barrier Energy](https://dl.acm.org/doi/10.1145/3643028)**,
*ACM Transactions on Graphics*, 2024.

This project serves as an excellent benchmark for conducting further research on GPU IPC,
enabling valuable comparisons to be made.

**Authors:** Kemeng Huang, Floyd M. Chitalu, Huancheng Lin, Taku Komura

**Source code contributor:** [Kemeng Huang](https://kemenghuang.github.io)

[![Watch the video](https://github.com/KemengHuang/GPU_IPC/blob/main/Assets/video1.png)](https://youtu.be/5rwp6AiHtrw)

---

## Requirements

- **Hardware:** NVIDIA GPU (CUDA-capable)
- **Platforms:** Windows, Linux
- **Toolchain:** CMake ≥ 3.18, a C++17 compiler, CUDA Toolkit

| Name     | Version | Usage             | Import         |
| -------- | ------- | ----------------- | -------------- |
| CUDA     | ≥ 11.0  | GPU programming   | system install |
| Eigen3   | 3.4.0   | matrix calculation| package        |
| freeglut | 3.4.0   | visualization     | package        |
| GLEW     | 2.2.0   | visualization     | package        |

## Build

### Linux

```bash
sudo apt install libglew-dev freeglut3-dev libeigen3-dev

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Windows

We use [vcpkg](https://github.com/microsoft/vcpkg) to manage dependencies.
The simplest way to let CMake detect vcpkg is to set the system environment variable
`CMAKE_TOOLCHAIN_FILE` to `(YOUR_VCPKG_FOLDER)/vcpkg/scripts/buildsystems/vcpkg.cmake`.

```bat
vcpkg install eigen3 freeglut glew

cmake -S . -B build
cmake --build build --config Release
```

> **GPU architecture:** `CMakeLists.txt` targets `CUDA_ARCHITECTURES 86` (Ampere, sm_86)
> by default. Edit that line to match your GPU
> (e.g. `89` for Ada/RTX 40xx, `90` for Hopper).

## Run

```bash
# Linux
./build/gipc
# Windows
build\Release\gipc.exe
```

A GLUT window opens and starts the default simulation scene (two stacked bunnies
loaded from `Assets/tetMesh/bunny2.msh`). Asset and output paths are compiled in
via the `GIPC_ASSETS_DIR` / `GIPC_OUTPUT_DIR` definitions, so the executable can
be launched from any working directory.

### Viewer controls

| Input        | Action                          |
| ------------ | ------------------------------- |
| mouse drag   | rotate view                     |
| `w` `s` `a` `d` `q` `e` | translate view         |
| `space`      | pause / resume simulation       |
| `k`          | toggle surface rendering        |
| `f`          | toggle BVH rendering            |
| `9`          | toggle surface export           |
| `/`          | toggle screenshot capture       |

## Configuration

Simulation parameters live in `Assets/scene/parameterSetting.txt` — material
properties (density, Young's modulus, Poisson ratio, friction), time step,
solver tolerances (`pcg_solver_threshold`, `Newton_solver_threshold`), the
relative collision distance threshold (`IPC_ralative_dHat`), etc. Edit the
text file and restart; no recompilation needed.

## Project structure

```
GPU_IPC/
├── GIPC.cu/.cuh          # core IPC solver: barrier gradient/Hessian, CCD,
│                         # line search, Newton main loop
├── PCG_SOLVER.cu/.cuh    # GPU PCG and MAS-PCG linear solvers
├── MASPreconditioner.*   # preconditioner assembly
├── mlbvh.cu/.cuh         # LBVH broad-phase collision detection
├── ACCD.cu/.cuh          # continuous collision detection
├── femEnergy.cu/.cuh     # FEM elasticity energy
├── gpu_eigen_libs.cuh    # header-only fixed-size matrix math (SVD, solver)
├── GIPC_PDerivative.cuh  # auto-generated analytic energy derivatives
├── gl_main.cpp           # GLUT viewer and scene setup
└── load_mesh.*           # mesh I/O
Assets/
├── scene/parameterSetting.txt   # simulation parameters
├── tetMesh/                     # volumetric meshes (.msh)
└── triMesh/                     # cloth meshes (.obj)
```

## BibTeX

Please cite the following paper if it helps.

```bibtex
@article{gipc2024,
author = {Huang, Kemeng and Chitalu, Floyd M. and Lin, Huancheng and Komura, Taku},
title = {GIPC: Fast and Stable Gauss-Newton Optimization of IPC Barrier Energy},
year = {2024},
issue_date = {April 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {43},
number = {2},
issn = {0730-0301},
url = {https://doi.org/10.1145/3643028},
doi = {10.1145/3643028},
abstract = {Barrier functions are crucial for maintaining an intersection- and inversion-free simulation trajectory but existing methods, which directly use distance can restrict implementation design and performance. We present an approach to rewriting the barrier function for arriving at an efficient and robust approximation of its Hessian. The key idea is to formulate a simplicial geometric measure of contact using mesh boundary elements, from which analytic eigensystems are derived and enhanced with filtering and stiffening terms that ensure robustness with respect to the convergence of a Project-Newton solver. A further advantage of our rewriting of the barrier function is that it naturally caters to the notorious case of nearly parallel edge-edge contacts for which we also present a novel analytic eigensystem. Our approach is thus well suited for standard second-order unconstrained optimization strategies for resolving contacts, minimizing nonlinear nonconvex functions where the Hessian may be indefinite. The efficiency of our eigensystems alone yields a 3\texttimes{} speedup over the standard Incremental Potential Contact (IPC) barrier formulation. We further apply our analytic proxy eigensystems to produce an entirely GPU-based implementation of IPC with significant further acceleration.},
journal = {ACM Trans. Graph.},
month = {mar},
articleno = {23},
numpages = {18},
keywords = {IPC, Barrier Hessian, eigen analysis, GPU}
}
```
