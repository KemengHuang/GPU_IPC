//
// mlbvh.cuh
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#pragma once
#ifndef _MLBVH_CUH_
#define _MLBVH_CUH_
#include <cstdint>
#include <cuda_runtime.h>
#include "device_buffer.h"
#include "device_launch_parameters.h"

struct AABB {
public:
    double3 upper;
    double3 lower;
    __host__ __device__  AABB();
    __host__ __device__  void combines(const double& x, const double& y, const double& z);
    __host__ __device__  void combines(const double& x, const double& y, const double& z, const double& xx, const double& yy, const double& zz);
    __host__ __device__  void combines(const AABB& aabb);
    __host__ __device__  double3 center();
};

struct Node {
public:
    uint32_t parent_idx;
    uint32_t left_idx;
    uint32_t right_idx;
    uint32_t element_idx;
};

class lbvh {
public:
    uint32_t vert_number = 0;
    double3* _vertexes = nullptr;
    gpuipc::DeviceBuffer<AABB> _bvs;
    gpuipc::DeviceBuffer<AABB> _tempLeafBox;
    gpuipc::DeviceBuffer<Node> _nodes;
    gpuipc::DeviceBuffer<uint64_t> _MChash;
    gpuipc::DeviceBuffer<uint32_t> _indices;
    gpuipc::DeviceBuffer<uint64_t> _sort_morton_codes;
    gpuipc::DeviceBuffer<uint32_t> _sort_indices;
    gpuipc::DeviceBuffer<uint32_t> _flags;
    AABB scene;
    int* _btype = nullptr;
public:
    lbvh() {}
    ~lbvh();
    void MALLOC_DEVICE_MEM(const int& number);
    void FREE_DEVICE_MEM();
    void sort_morton_codes(int number);
    //void Construct();
};


class lbvh_f : public lbvh {
public:
    uint32_t face_number = 0;
    uint3* _faces = nullptr;
    uint32_t* _surfVerts = nullptr;
public:
    void init(int* _btype, double3* _mVerts, uint3* _mFaces, uint32_t* _mSurfVert, const int& faceNum, const int& vertNum);
    double Construct();
    AABB* getSceneSize();
    double ConstructFullCCD(const double3* moveDir, const double& alpha);
    void SelfCollitionDetect(double dHat, int4* collisionPairs, int4* ccdCollisionPairs, uint32_t* pairCounts, int* matrixIndices, uint32_t pairCapacity);
    void SelfCollitionFullDetect(double dHat, const double3* moveDir, const double& alpha, int4* ccdCollisionPairs, uint32_t* pairCount, uint32_t pairCapacity);
};

class lbvh_e : public lbvh{
public:
    double3* _rest_vertexes = nullptr;
    uint32_t edge_number = 0;
    uint2* _edges = nullptr;
public:
    void init(int* _btype, double3* _mVerts, double3* _rest_vertexes, uint2* _mEdges, const int& edgeNum, const int& vertNum);
    double Construct();
    double ConstructFullCCD(const double3* moveDir, const double& alpha);
    void SelfCollitionDetect(double dHat, int4* collisionPairs, int4* ccdCollisionPairs, uint32_t* pairCounts, int* matrixIndices, uint32_t pairCapacity);
    void SelfCollitionFullDetect(double dHat, const double3* moveDir, const double& alpha, int4* ccdCollisionPairs, uint32_t* pairCount, uint32_t pairCapacity);
};

__device__
void _d_PP(const double3& v0, const double3& v1, double& d);

__device__
void _d_PT(const double3& v0, const double3& v1, const double3& v2, const double3& v3, double& d);

__device__
void _d_PE(const double3& v0, const double3& v1, const double3& v2, double& d);

__device__
void _d_EE(const double3& v0, const double3& v1, const double3& v2, const double3& v3, double& d);

__device__
void _d_EEParallel(const double3& v0, const double3& v1, const double3& v2, const double3& v3, double& d);

__device__
double _compute_epx(const double3& v0, const double3& v1, const double3& v2, const double3& v3);

#endif
