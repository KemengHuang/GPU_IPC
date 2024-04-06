# GPU IPC


DESCRIPTION
===========

This is the first fully GPU optimized IPC framework and the source code of the paper: [GIPC: Fast and Stable Gauss-Newton Optimization of IPC Barrier Energy](https://dl.acm.org/doi/10.1145/3643028), **ACM Transaction on Graphics, 2024**. This project serves as an excellent benchmark for conducting further research on GPU IPC, enabling valuable comparisons to be made.

authors: Kemeng Huang, Floyd M. Chitalu, Huancheng Lin, Taku Komura

Source code contributor: [Kemeng Huang](https://kemenghuang.github.io)

**Note: this software is released under the MPLv2.0 license. For commercial use, please email the authors for negotiation.**

## video 1
[![Watch the video](https://github.com/KemengHuang/GPU_IPC/blob/main/Assets/video1.png)](https://www.youtube.com/watch?v=zJ0_zsU47h4&t=4s)

## video 2
[![Watch the video](https://github.com/KemengHuang/GPU_IPC/blob/main/Assets/video2.png)](https://www.youtube.com/watch?v=GE39Ar1uH9g)

## BibTex 

Please cite the following paper if it helps. 

```
@article{10.1145/3643028,
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


Requirements
============

Hardware requirements: Nvidia GPUs

Support platforms: Windows, Linux 

## Dependencies

| Name                                   | Version | Usage                                               | Import         |
| -------------------------------------- | ------- | --------------------------------------------------- | -------------- |
| cuda                                   | >=11.0  | GPU programming                                     | system install |
| eigen3                                 | 3.4.0   | matrix calculation                                  | package        |
| freeglut                               | 3.4.0   | visualization                                       | package        |
| glew                                   | 2.2.0#3 | visualization                                       | package        |

### linux

We use CMake to build the project.

```bash
sudo apt install libglew-dev freeglut3-dev libeigen3-dev
```


### Windows
We use [vcpkg](https://github.com/microsoft/vcpkg) to manage the libraries we need and use CMake to build the project. The simplest way to let CMake detect vcpkg is to set the system environment variable `CMAKE_TOOLCHAIN_FILE` to `(YOUR_VCPKG_PARENT_FOLDER)/vcpkg/scripts/buildsystems/vcpkg.cmake`

```shell
vcpkg install eigen3 freeglut glew
```

EXTERNAL CREDITS
================

This work utilizes the following code, which have been included here for convenience:
Copyrights are retained by the original authors.

zpc https://github.com/zenustech/zpc
