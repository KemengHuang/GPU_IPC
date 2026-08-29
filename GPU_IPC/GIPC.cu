//
// GIPC.cu
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include "GIPC.cuh"
#include "cuda_tools.h"
#include "GIPC_PDerivative.cuh"
#include "fem_parameters.h"
#include "ACCD.cuh"
#include "femEnergy.cuh"
#include "cuda_cub_wrappers.h"
#include "FrictionUtils.cuh"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include "Eigen/Eigen"
#include "device_utils.h"
using namespace Eigen;

#define RANK 2
#define NEWF
#define MAKEPD2
#define OLDBARRIER2

__global__ void fill_sort_indices(uint32_t* indices, int number)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < number)
        indices[i] = static_cast<uint32_t>(i);
}
template <class F>
__device__ __host__ inline F __m_min(F a, F b)
{
    return a > b ? b : a;
}


template <class F>
__device__ __host__ inline F __m_max(F a, F b)
{
    return a > b ? a : b;
}

__device__ __host__ inline uint32_t expand_bits(std::uint32_t v) noexcept
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ __host__ inline uint32_t hash_code(
    int type, double x, double y, double z, double resolution = 1024) noexcept
{
    x = __m_min(__m_max(x * resolution, 0.0), resolution - 1.0);
    y = __m_min(__m_max(y * resolution, 0.0), resolution - 1.0);
    z = __m_min(__m_max(z * resolution, 0.0), resolution - 1.0);


    //
    if(type == -1)
    {
        const uint32_t xx     = expand_bits(static_cast<uint32_t>(x));
        const uint32_t yy     = expand_bits(static_cast<uint32_t>(y));
        const uint32_t zz     = expand_bits(static_cast<uint32_t>(z));
        std::uint32_t  mchash = ((xx << 2) + (yy << 1) + zz);

        return mchash;
    }
    else if(type == 0)
    {
        return (((static_cast<uint32_t>(z) * 1024) + static_cast<uint32_t>(y)) * 1024)
               + static_cast<uint32_t>(x);
    }
    else if(type == 1)
    {
        return (((static_cast<uint32_t>(y) * 1024) + static_cast<uint32_t>(z)) * 1024)
               + static_cast<uint32_t>(x);
    }
    else if(type == 2)
    {
        return (((static_cast<uint32_t>(x) * 1024) + static_cast<uint32_t>(z)) * 1024)
               + static_cast<uint32_t>(y);
    }
    else if(type == 3)
    {
        return (((static_cast<uint32_t>(z) * 1024) + static_cast<uint32_t>(x)) * 1024)
               + static_cast<uint32_t>(y);
    }
    else if(type == 4)
    {
        return (((static_cast<uint32_t>(y) * 1024) + static_cast<uint32_t>(x)) * 1024)
               + static_cast<uint32_t>(z);
    }
    else
    {
        return (((static_cast<uint32_t>(x) * 1024) + static_cast<uint32_t>(y)) * 1024)
               + static_cast<uint32_t>(z);
    }
    //std::uint32_t mchash = (((static_cast<std::uint32_t>(z) * 1024) + static_cast<std::uint32_t>(y)) * 1024) + static_cast<std::uint32_t>(x);//((xx << 2) + (yy << 1) + zz);
    //return mchash;
}

// =============================================================================
// Block-reduction device helpers
//
// Warp-shuffle reduction of `temp` across the whole block; the per-block
// partial result is written to out[blockIdx.x]. `number` is the number of
// active elements in the (possibly partial) last block.
//
// This is the exact epilogue previously copy-pasted into every reduction
// kernel in this file (sum / max / max-double2 variants); behavior unchanged.
// =============================================================================

template <typename Combine>
__device__ __forceinline__ void __blockReduce(double temp, double* tep, double* out, int number, Combine combine)
{
    int idof    = blockIdx.x * blockDim.x;
    int warpTid = threadIdx.x % 32;
    int warpId  = (threadIdx.x >> 5);
    int warpNum;
    if(blockIdx.x == gridDim.x - 1)
    {
        warpNum = ((number - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    const int warp_begin = idof + warpId * warpSize;
    const int valid_lanes = min(warpSize, max(number - warp_begin, 0));
    const unsigned int warp_mask =
        gipc::WARP_MASK_FROM_COUNT(static_cast<unsigned int>(valid_lanes));
    for(int i = 1; i < 32; i = (i << 1))
    {
        const double other = gipc::WARP_SHFL_DOWN(temp, i, warp_mask);
        if(warpTid + i < valid_lanes)
            temp = combine(temp, other);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    gipc::SYNC_THREADS();
    if(threadIdx.x < warpNum)
    {
        if(warpNum > 1)
        {
            temp = tep[threadIdx.x];
            const unsigned int aggregate_mask =
                gipc::WARP_MASK_FROM_COUNT(static_cast<unsigned int>(warpNum));
            for(int i = 1; i < warpNum; i = (i << 1))
            {
                const double other =
                    gipc::WARP_SHFL_DOWN(temp, i, aggregate_mask);
                if(threadIdx.x + i < warpNum)
                    temp = combine(temp, other);
            }
        }
        if(threadIdx.x == 0)
        {
            out[blockIdx.x] = temp;
        }
    }
}

struct __reduce_sum {
    __device__ __forceinline__ double operator()(double a, double b) const { return a + b; }
};

struct __reduce_max {
    __device__ __forceinline__ double operator()(double a, double b) const { return __m_max(a, b); }
};

__device__ __forceinline__ void __blockReduceSum(double temp, double* tep, double* out, int number)
{
    __blockReduce(temp, tep, out, number, __reduce_sum());
}

__device__ __forceinline__ void __blockReduceMax(double temp, double* tep, double* out, int number)
{
    __blockReduce(temp, tep, out, number, __reduce_max());
}

__device__ __forceinline__ void __blockReduceMax2(double2 temp, double2* sdata, double2* out, int number)
{
    int idof    = blockIdx.x * blockDim.x;
    int warpTid = threadIdx.x % 32;
    int warpId  = (threadIdx.x >> 5);
    int warpNum;
    if(blockIdx.x == gridDim.x - 1)
    {
        warpNum = ((number - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    const int warp_begin = idof + warpId * warpSize;
    const int valid_lanes = min(warpSize, max(number - warp_begin, 0));
    const unsigned int warp_mask =
        gipc::WARP_MASK_FROM_COUNT(static_cast<unsigned int>(valid_lanes));
    for(int i = 1; i < 32; i = (i << 1))
    {
        const double tempMin = gipc::WARP_SHFL_DOWN(temp.x, i, warp_mask);
        const double tempMax = gipc::WARP_SHFL_DOWN(temp.y, i, warp_mask);
        if(warpTid + i < valid_lanes)
        {
            temp.x = __m_max(temp.x, tempMin);
            temp.y = __m_max(temp.y, tempMax);
        }
    }
    if(warpTid == 0)
    {
        sdata[warpId] = temp;
    }
    gipc::SYNC_THREADS();
    if(threadIdx.x < warpNum)
    {
        if(warpNum > 1)
        {
            temp = sdata[threadIdx.x];
            const unsigned int aggregate_mask =
                gipc::WARP_MASK_FROM_COUNT(static_cast<unsigned int>(warpNum));
            for(int i = 1; i < warpNum; i = (i << 1))
            {
                const double tempMin =
                    gipc::WARP_SHFL_DOWN(temp.x, i, aggregate_mask);
                const double tempMax =
                    gipc::WARP_SHFL_DOWN(temp.y, i, aggregate_mask);
                if(threadIdx.x + i < warpNum)
                {
                    temp.x = __m_max(temp.x, tempMin);
                    temp.y = __m_max(temp.y, tempMax);
                }
            }
        }
        if(threadIdx.x == 0)
        {
            out[blockIdx.x] = temp;
        }
    }
}

// =============================================================================
// Streaming low-rank Hessian stores
//
// Every barrier Hessian in _calBarrierGradientAndHessian is low-rank:
//   H = A * (sum_k lambdas[k] * q_k * q_k^T) * A^T = sum_k lambdas[k] * w_k * w_k^T
// with w_k = A * q_k (a 12-/9-vector).  Each output element is accumulated in
// registers and written straight to global memory, so no 9x9 / 12x12 local
// matrix is ever materialized (previously this cost a 2128B stack frame per
// thread).  Mathematically identical to the former triple-matrix product.
// A rank component with lambdas[k] == 0 contributes nothing (equivalent to
// skipping it, as the old makePD code did for non-positive eigenvalues).
// =============================================================================

__device__ __forceinline__ void __storeHessian12Rank1(__GEIGEN__::Matrix12x12d* out,
                                                      int                        Hidx,
                                                      double                     lambda,
                                                      const __GEIGEN__::Vector12& w)
{
#pragma unroll
    for(int i = 0; i < 12; i++)
    {
        double li = lambda * w.v[i];
#pragma unroll
        for(int j = 0; j < 12; j++)
        {
            out[Hidx].m[i][j] = li * w.v[j];
        }
    }
}

__device__ __forceinline__ void __storeHessian12RankN(__GEIGEN__::Matrix12x12d* out,
                                                      int                        Hidx,
                                                      double                     l0,
                                                      double                     l1,
                                                      double                     l2,
                                                      double                     l3,
                                                      const __GEIGEN__::Vector12& w0,
                                                      const __GEIGEN__::Vector12& w1,
                                                      const __GEIGEN__::Vector12& w2,
                                                      const __GEIGEN__::Vector12& w3)
{
#pragma unroll
    for(int i = 0; i < 12; i++)
    {
        double a0 = l0 * w0.v[i], a1 = l1 * w1.v[i], a2 = l2 * w2.v[i], a3 = l3 * w3.v[i];
#pragma unroll
        for(int j = 0; j < 12; j++)
        {
            out[Hidx].m[i][j] = a0 * w0.v[j] + a1 * w1.v[j] + a2 * w2.v[j] + a3 * w3.v[j];
        }
    }
}

__device__ __forceinline__ void __storeHessian9Rank1(__GEIGEN__::Matrix9x9d*  out,
                                                     int                      Hidx,
                                                     double                   lambda,
                                                     const __GEIGEN__::Vector9& w)
{
#pragma unroll
    for(int i = 0; i < 9; i++)
    {
        double li = lambda * w.v[i];
#pragma unroll
        for(int j = 0; j < 9; j++)
        {
            out[Hidx].m[i][j] = li * w.v[j];
        }
    }
}

__global__ void _calcTetMChash(uint64_t*       _MChash,
                               const double3*  _vertexes,
                               uint4*          tets,
                               const AABB*     _MaxBv,
                               const uint32_t* sortMapVertIndex,
                               int             number)
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= number)
        return;

    tets[idx].x = sortMapVertIndex[tets[idx].x];
    tets[idx].y = sortMapVertIndex[tets[idx].y];
    tets[idx].z = sortMapVertIndex[tets[idx].z];
    tets[idx].w = sortMapVertIndex[tets[idx].w];

    double3 SceneSize = make_double3((*_MaxBv).upper.x - (*_MaxBv).lower.x,
                                     (*_MaxBv).upper.y - (*_MaxBv).lower.y,
                                     (*_MaxBv).upper.z - (*_MaxBv).lower.z);
    double3 centerP   = __GEIGEN__::__s_vec_multiply(
        __GEIGEN__::__add(
            __GEIGEN__::__add(_vertexes[tets[idx].x], _vertexes[tets[idx].y]),
            __GEIGEN__::__add(_vertexes[tets[idx].z], _vertexes[tets[idx].w])),
        0.25);
    double3 offset = make_double3(centerP.x - (*_MaxBv).lower.x,
                                  centerP.y - (*_MaxBv).lower.y,
                                  centerP.z - (*_MaxBv).lower.z);

    int type = 0;
    if(SceneSize.x > SceneSize.y && SceneSize.y > SceneSize.z)
    {
        type = 0;
    }
    else if(SceneSize.x > SceneSize.z && SceneSize.z > SceneSize.y)
    {
        type = 1;
    }
    else if(SceneSize.y > SceneSize.z && SceneSize.z > SceneSize.x)
    {
        type = 2;
    }
    else if(SceneSize.y > SceneSize.x && SceneSize.x > SceneSize.z)
    {
        type = 3;
    }
    else if(SceneSize.z > SceneSize.x && SceneSize.x > SceneSize.y)
    {
        type = 4;
    }
    else
    {
        type = 5;
    }

    //printf("%d   %f     %f     %f\n", offset.x, offset.y, offset.z);
    uint64_t mc32 = hash_code(type,
                              offset.x / SceneSize.x,
                              offset.y / SceneSize.y,
                              offset.z / SceneSize.z);
    uint64_t mc64 = ((mc32 << 32) | idx);
    //printf("morton code %d\n", mc64);
    _MChash[idx] = mc64;
}

__global__ void _updateTopology(uint4*          tets,
                                uint3*          tris,
                                const uint32_t* sortMapVertIndex,
                                int             traNumber,
                                int             triNumber)
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < traNumber)
    {

        tets[idx].x = sortMapVertIndex[tets[idx].x];
        tets[idx].y = sortMapVertIndex[tets[idx].y];
        tets[idx].z = sortMapVertIndex[tets[idx].z];
        tets[idx].w = sortMapVertIndex[tets[idx].w];
    }
    if(idx < triNumber)
    {
        tris[idx].x = sortMapVertIndex[tris[idx].x];
        tris[idx].y = sortMapVertIndex[tris[idx].y];
        tris[idx].z = sortMapVertIndex[tris[idx].z];
    }
}


__global__ void _updateVertexes(double3*                      o_vertexes,
                                const double3*                _vertexes,
                                double*                       tempM,
                                const double*                 mass,
                                __GEIGEN__::Matrix3x3d*       tempCons,
                                int*                          tempBtype,
                                const __GEIGEN__::Matrix3x3d* cons,
                                const int*                    bType,
                                const uint32_t*               sortIndex,
                                uint32_t*                     sortMapIndex,
                                int                           number)
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= number)
        return;
    o_vertexes[idx]              = _vertexes[sortIndex[idx]];
    tempM[idx]                   = mass[sortIndex[idx]];
    tempCons[idx]                = cons[sortIndex[idx]];
    sortMapIndex[sortIndex[idx]] = idx;
    tempBtype[idx]               = bType[sortIndex[idx]];
    //printf("original idx: %d        new idx: %d\n", sortIndex[idx], idx);
}

__global__ void _updateTetrahedras(uint4*                        o_tetrahedras,
                                   uint4*                        tetrahedras,
                                   double*                       tempV,
                                   const double*                 volum,
                                   __GEIGEN__::Matrix3x3d*       tempDmInverse,
                                   const __GEIGEN__::Matrix3x3d* dmInverse,
                                   const uint32_t*               sortTetIndex,
                                   const uint32_t* sortMapVertIndex,
                                   int             number)
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= number)
        return;
    //tetrahedras[idx].x = sortMapVertIndex[tetrahedras[idx].x];
    //tetrahedras[idx].y = sortMapVertIndex[tetrahedras[idx].y];
    //tetrahedras[idx].z = sortMapVertIndex[tetrahedras[idx].z];
    //tetrahedras[idx].w = sortMapVertIndex[tetrahedras[idx].w];
    o_tetrahedras[idx] = tetrahedras[sortTetIndex[idx]];
    tempV[idx]         = volum[sortTetIndex[idx]];
    tempDmInverse[idx] = dmInverse[sortTetIndex[idx]];
}

__global__ void _calcVertMChash(uint64_t* _MChash, const double3* _vertexes, const AABB* _MaxBv, int number)
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= number)
        return;
    double3 SceneSize = make_double3((*_MaxBv).upper.x - (*_MaxBv).lower.x,
                                     (*_MaxBv).upper.y - (*_MaxBv).lower.y,
                                     (*_MaxBv).upper.z - (*_MaxBv).lower.z);
    double3 centerP   = _vertexes[idx];
    double3 offset    = make_double3(centerP.x - (*_MaxBv).lower.x,
                                  centerP.y - (*_MaxBv).lower.y,
                                  centerP.z - (*_MaxBv).lower.z);
    int     type      = -1;
    if(type >= 0)
    {
        if(SceneSize.x > SceneSize.y && SceneSize.y > SceneSize.z)
        {
            type = 0;
        }
        else if(SceneSize.x > SceneSize.z && SceneSize.z > SceneSize.y)
        {
            type = 1;
        }
        else if(SceneSize.y > SceneSize.z && SceneSize.z > SceneSize.x)
        {
            type = 2;
        }
        else if(SceneSize.y > SceneSize.x && SceneSize.x > SceneSize.z)
        {
            type = 3;
        }
        else if(SceneSize.z > SceneSize.x && SceneSize.x > SceneSize.y)
        {
            type = 4;
        }
        else
        {
            type = 5;
        }
    }

    //printf("minSize %f     %f     %f\n", SceneSize.x, SceneSize.y, SceneSize.z);
    uint64_t mc32 = hash_code(type,
                              offset.x / SceneSize.x,
                              offset.y / SceneSize.y,
                              offset.z / SceneSize.z);
    uint64_t mc64 = ((mc32 << 32) | idx);
    //printf("morton code %lld\n", mc64);
    _MChash[idx] = mc64;
}

__global__ void _reduct_max_double3_to_double(const double3* _double3Dim, double* _double1Dim, int number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    double3 tempMove = _double3Dim[idx];

    double temp =
        __m_max(__m_max(abs(tempMove.x), abs(tempMove.y)), abs(tempMove.z));

    __blockReduceMax(temp, tep, _double1Dim, number);
}


__global__ void _reduct_M_double2(double2* _double2Dim, int number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double2 sdata[];

    if(idx >= number)
        return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    double2 temp = _double2Dim[idx];

    gipc::THREAD_FENCE();


    __blockReduceMax2(temp, sdata, _double2Dim, number);
}


__global__ void _reduct_max_double(double* _double1Dim, int number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    double temp = _double1Dim[idx];

    gipc::THREAD_FENCE();


    __blockReduceMax(temp, tep, _double1Dim, number);
}


__device__ double __cal_Barrier_energy(const double3* _vertexes,
                                       const double3* _rest_vertexes,
                                       int4           MMCVIDI,
                                       double         _Kappa,
                                       double         _dHat)
{
    double dHat_sqrt = sqrt(_dHat);
    double dHat      = _dHat;
    double Kappa     = _Kappa;
    if(MMCVIDI.x >= 0)
    {
        if(MMCVIDI.w >= 0)
        {
            double dis;
            _d_EE(_vertexes[MMCVIDI.x],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            double I5 = dis / dHat;

            double lenE = (dis - dHat);
#if(RANK == 1)
            return -Kappa * lenE * lenE * log(I5);
#elif(RANK == 2)
            return Kappa * lenE * lenE * log(I5) * log(I5);
#elif(RANK == 3)
            return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5);
#elif(RANK == 4)
            return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5);
#elif(RANK == 5)
            return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5) * log(I5);
#elif(RANK == 6)
            return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5)
                   * log(I5) * log(I5);
#endif
        }
        else
        {
            //return 0;
            MMCVIDI.w = -MMCVIDI.w - 1;
            double3 v0 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
            double3 v1 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z]);
            double c  = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1)) /*/ __GEIGEN__::__norm(v0)*/;
            double I1 = c * c;
            if(I1 == 0)
                return 0;
            double dis;
            _d_EE(_vertexes[MMCVIDI.x],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            double I2    = dis / dHat;
            double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                        _rest_vertexes[MMCVIDI.y],
                                        _rest_vertexes[MMCVIDI.z],
                                        _rest_vertexes[MMCVIDI.w]);
#if(RANK == 1)
            double Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                            * -(dHat - dHat * I2) * (dHat - dHat * I2) * log(I2);
#elif(RANK == 2)
            double Energy =
                Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2);
#elif(RANK == 4)
            double Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                            * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2)
                            * log(I2) * log(I2) * log(I2);
#elif(RANK == 6)
            double Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                            * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2)
                            * log(I2) * log(I2) * log(I2) * log(I2) * log(I2);
#endif
            if(Energy < 0)
                printf("I am pee\n");
            return Energy;
        }
    }
    else
    {
        int v0I = -MMCVIDI.x - 1;
        if(MMCVIDI.z < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.z = -MMCVIDI.z - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;
                MMCVIDI.x = v0I;

                double3 v0 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.x]);
                double3 v1 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.y]);
                double c  = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1)) /*/ __GEIGEN__::__norm(v0)*/;
                double I1 = c * c;
                if(I1 == 0)
                    return 0;
                double dis;
                _d_PP(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], dis);
                double I2    = dis / dHat;
                double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                            _rest_vertexes[MMCVIDI.z],
                                            _rest_vertexes[MMCVIDI.y],
                                            _rest_vertexes[MMCVIDI.w]);
#if(RANK == 1)
                double Energy =
                    Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                    * -(dHat - dHat * I2) * (dHat - dHat * I2) * log(I2);
#elif(RANK == 2)
                double Energy =
                    Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                    * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2);
#elif(RANK == 4)
                double Energy =
                    Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                    * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2)
                    * log(I2) * log(I2) * log(I2);
#elif(RANK == 6)
                double Energy =
                    Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                    * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2)
                    * log(I2) * log(I2) * log(I2) * log(I2) * log(I2);
#endif
                if(Energy < 0)
                    printf("I am pp\n");
                return Energy;
            }
            else
            {
                double dis;
                _d_PP(_vertexes[v0I], _vertexes[MMCVIDI.y], dis);
                double I5 = dis / dHat;

                double lenE = (dis - dHat);
#if(RANK == 1)
                return -Kappa * lenE * lenE * log(I5);
#elif(RANK == 2)
                return Kappa * lenE * lenE * log(I5) * log(I5);
#elif(RANK == 3)
                return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5);
#elif(RANK == 4)
                return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5);
#elif(RANK == 5)
                return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5)
                       * log(I5) * log(I5);
#elif(RANK == 6)
                return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5)
                       * log(I5) * log(I5) * log(I5);
#endif
            }
        }
        else if(MMCVIDI.w < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                //MMCVIDI.z = -MMCVIDI.z - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;
                MMCVIDI.x = v0I;

                double3 v0 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
                double3 v1 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]);
                double c  = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1)) /*/ __GEIGEN__::__norm(v0)*/;
                double I1 = c * c;
                if(I1 == 0)
                    return 0;
                double dis;
                _d_PE(_vertexes[MMCVIDI.x],
                      _vertexes[MMCVIDI.y],
                      _vertexes[MMCVIDI.z],
                      dis);
                double I2    = dis / dHat;
                double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                            _rest_vertexes[MMCVIDI.w],
                                            _rest_vertexes[MMCVIDI.y],
                                            _rest_vertexes[MMCVIDI.z]);
#if(RANK == 1)
                double Energy =
                    Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                    * -(dHat - dHat * I2) * (dHat - dHat * I2) * log(I2);
#elif(RANK == 2)
                double Energy =
                    Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                    * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2);
#elif(RANK == 4)
                double Energy =
                    Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                    * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2)
                    * log(I2) * log(I2) * log(I2);
#elif(RANK == 6)
                double Energy =
                    Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1)
                    * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2)
                    * log(I2) * log(I2) * log(I2) * log(I2) * log(I2);
#endif
                if(Energy < 0)
                    printf("I am ppe\n");
                return Energy;
            }
            else
            {
                double dis;
                _d_PE(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
                double I5 = dis / dHat;

                double lenE = (dis - dHat);
#if(RANK == 1)
                return -Kappa * lenE * lenE * log(I5);
#elif(RANK == 2)
                return Kappa * lenE * lenE * log(I5) * log(I5);
#elif(RANK == 3)
                return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5);
#elif(RANK == 4)
                return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5);
#elif(RANK == 5)
                return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5)
                       * log(I5) * log(I5);
#elif(RANK == 6)
                return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5)
                       * log(I5) * log(I5) * log(I5);
#endif
            }
        }
        else
        {
            double dis;
            _d_PT(_vertexes[v0I],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            double I5 = dis / dHat;

            double lenE = (dis - dHat);
#if(RANK == 1)
            return -Kappa * lenE * lenE * log(I5);
#elif(RANK == 2)
            return Kappa * lenE * lenE * log(I5) * log(I5);
#elif(RANK == 3)
            return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5);
#elif(RANK == 4)
            return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5);
#elif(RANK == 5)
            return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5) * log(I5);
#elif(RANK == 6)
            return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5)
                   * log(I5) * log(I5);
#endif
        }
    }
}

__device__ bool segTriIntersect(const double3& ve0,
                                const double3& ve1,
                                const double3& vt0,
                                const double3& vt1,
                                const double3& vt2)
{

    //printf("check for tri and lines\n");

    __GEIGEN__::Matrix3x3d coefMtr;
    double3                col0 = __GEIGEN__::__minus(vt1, vt0);
    double3                col1 = __GEIGEN__::__minus(vt2, vt0);
    double3                col2 = __GEIGEN__::__minus(ve0, ve1);

    __GEIGEN__::__set_Mat_val_column(coefMtr, col0, col1, col2);

    double3 n = __GEIGEN__::__v_vec_cross(col0, col1);
    if(__GEIGEN__::__v_vec_dot(n, __GEIGEN__::__minus(ve0, vt0))
           * __GEIGEN__::__v_vec_dot(n, __GEIGEN__::__minus(ve1, vt0))
       > 0)
    {
        return false;
    }

    double det = __GEIGEN__::__Determiant(coefMtr);

    if(abs(det) < 1e-20)
    {
        return false;
    }

    __GEIGEN__::Matrix3x3d D1, D2, D3;
    double3                b = __GEIGEN__::__minus(ve0, vt0);

    __GEIGEN__::__set_Mat_val_column(D1, b, col1, col2);
    __GEIGEN__::__set_Mat_val_column(D2, col0, b, col2);
    __GEIGEN__::__set_Mat_val_column(D3, col0, col1, b);

    double uvt[3];
    uvt[0] = __GEIGEN__::__Determiant(D1) / det;
    uvt[1] = __GEIGEN__::__Determiant(D2) / det;
    uvt[2] = __GEIGEN__::__Determiant(D3) / det;

    if(uvt[0] >= 0.0 && uvt[1] >= 0.0 && uvt[0] + uvt[1] <= 1.0 && uvt[2] >= 0.0
       && uvt[2] <= 1.0)
    {
        return true;
    }
    else
    {
        return false;
    }
}

__device__ __host__ inline bool _overlap(const AABB& lhs, const AABB& rhs, const double& gapL) noexcept
{
    if((rhs.lower.x - lhs.upper.x) >= gapL || (lhs.lower.x - rhs.upper.x) >= gapL)
        return false;
    if((rhs.lower.y - lhs.upper.y) >= gapL || (lhs.lower.y - rhs.upper.y) >= gapL)
        return false;
    if((rhs.lower.z - lhs.upper.z) >= gapL || (lhs.lower.z - rhs.upper.z) >= gapL)
        return false;
    return true;
}

__device__ double _selfConstraintVal(const double3* vertexes, const int4& active)
{
    double val;
    if(active.x >= 0)
    {
        if(active.w >= 0)
        {
            _d_EE(vertexes[active.x],
                  vertexes[active.y],
                  vertexes[active.z],
                  vertexes[active.w],
                  val);
        }
        else
        {
            _d_EE(vertexes[active.x],
                  vertexes[active.y],
                  vertexes[active.z],
                  vertexes[-active.w - 1],
                  val);
        }
    }
    else
    {
        if(active.z < 0)
        {
            if(active.y < 0)
            {
                _d_PP(vertexes[-active.x - 1], vertexes[-active.y - 1], val);
            }
            else
            {
                _d_PP(vertexes[-active.x - 1], vertexes[active.y], val);
            }
        }
        else if(active.w < 0)
        {
            if(active.y < 0)
            {
                _d_PE(vertexes[-active.x - 1],
                      vertexes[-active.y - 1],
                      vertexes[active.z],
                      val);
            }
            else
            {
                _d_PE(
                    vertexes[-active.x - 1], vertexes[active.y], vertexes[active.z], val);
            }
        }
        else
        {
            _d_PT(vertexes[-active.x - 1],
                  vertexes[active.y],
                  vertexes[active.z],
                  vertexes[active.w],
                  val);
        }
    }
    return val;
}

__device__ double _computeInjectiveStepSize_3d(const double3*  verts,
                                               const double3*  mv,
                                               const uint32_t& v0,
                                               const uint32_t& v1,
                                               const uint32_t& v2,
                                               const uint32_t& v3,
                                               double          ratio,
                                               double          errorRate)
{

    double x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4;
    double p1, p2, p3, p4, q1, q2, q3, q4, r1, r2, r3, r4;
    double a, b, c, d, t;


    x1 = verts[v0].x;
    x2 = verts[v1].x;
    x3 = verts[v2].x;
    x4 = verts[v3].x;

    y1 = verts[v0].y;
    y2 = verts[v1].y;
    y3 = verts[v2].y;
    y4 = verts[v3].y;

    z1 = verts[v0].z;
    z2 = verts[v1].z;
    z3 = verts[v2].z;
    z4 = verts[v3].z;

    p1 = -mv[v0].x;
    p2 = -mv[v1].x;
    p3 = -mv[v2].x;
    p4 = -mv[v3].x;

    q1 = -mv[v0].y;
    q2 = -mv[v1].y;
    q3 = -mv[v2].y;
    q4 = -mv[v3].y;

    r1 = -mv[v0].z;
    r2 = -mv[v1].z;
    r3 = -mv[v2].z;
    r4 = -mv[v3].z;

    a = -p1 * q2 * r3 + p1 * r2 * q3 + q1 * p2 * r3 - q1 * r2 * p3 - r1 * p2 * q3
        + r1 * q2 * p3 + p1 * q2 * r4 - p1 * r2 * q4 - q1 * p2 * r4 + q1 * r2 * p4
        + r1 * p2 * q4 - r1 * q2 * p4 - p1 * q3 * r4 + p1 * r3 * q4 + q1 * p3 * r4
        - q1 * r3 * p4 - r1 * p3 * q4 + r1 * q3 * p4 + p2 * q3 * r4 - p2 * r3 * q4
        - q2 * p3 * r4 + q2 * r3 * p4 + r2 * p3 * q4 - r2 * q3 * p4;
    b = -x1 * q2 * r3 + x1 * r2 * q3 + y1 * p2 * r3 - y1 * r2 * p3 - z1 * p2 * q3
        + z1 * q2 * p3 + x2 * q1 * r3 - x2 * r1 * q3 - y2 * p1 * r3
        + y2 * r1 * p3 + z2 * p1 * q3 - z2 * q1 * p3 - x3 * q1 * r2
        + x3 * r1 * q2 + y3 * p1 * r2 - y3 * r1 * p2 - z3 * p1 * q2 + z3 * q1 * p2
        + x1 * q2 * r4 - x1 * r2 * q4 - y1 * p2 * r4 + y1 * r2 * p4 + z1 * p2 * q4
        - z1 * q2 * p4 - x2 * q1 * r4 + x2 * r1 * q4 + y2 * p1 * r4 - y2 * r1 * p4
        - z2 * p1 * q4 + z2 * q1 * p4 + x4 * q1 * r2 - x4 * r1 * q2 - y4 * p1 * r2
        + y4 * r1 * p2 + z4 * p1 * q2 - z4 * q1 * p2 - x1 * q3 * r4 + x1 * r3 * q4
        + y1 * p3 * r4 - y1 * r3 * p4 - z1 * p3 * q4 + z1 * q3 * p4 + x3 * q1 * r4
        - x3 * r1 * q4 - y3 * p1 * r4 + y3 * r1 * p4 + z3 * p1 * q4 - z3 * q1 * p4
        - x4 * q1 * r3 + x4 * r1 * q3 + y4 * p1 * r3 - y4 * r1 * p3 - z4 * p1 * q3
        + z4 * q1 * p3 + x2 * q3 * r4 - x2 * r3 * q4 - y2 * p3 * r4 + y2 * r3 * p4
        + z2 * p3 * q4 - z2 * q3 * p4 - x3 * q2 * r4 + x3 * r2 * q4 + y3 * p2 * r4
        - y3 * r2 * p4 - z3 * p2 * q4 + z3 * q2 * p4 + x4 * q2 * r3 - x4 * r2 * q3
        - y4 * p2 * r3 + y4 * r2 * p3 + z4 * p2 * q3 - z4 * q2 * p3;
    c = -x1 * y2 * r3 + x1 * z2 * q3 + x1 * y3 * r2 - x1 * z3 * q2 + y1 * x2 * r3
        - y1 * z2 * p3 - y1 * x3 * r2 + y1 * z3 * p2 - z1 * x2 * q3
        + z1 * y2 * p3 + z1 * x3 * q2 - z1 * y3 * p2 - x2 * y3 * r1
        + x2 * z3 * q1 + y2 * x3 * r1 - y2 * z3 * p1 - z2 * x3 * q1 + z2 * y3 * p1
        + x1 * y2 * r4 - x1 * z2 * q4 - x1 * y4 * r2 + x1 * z4 * q2 - y1 * x2 * r4
        + y1 * z2 * p4 + y1 * x4 * r2 - y1 * z4 * p2 + z1 * x2 * q4 - z1 * y2 * p4
        - z1 * x4 * q2 + z1 * y4 * p2 + x2 * y4 * r1 - x2 * z4 * q1 - y2 * x4 * r1
        + y2 * z4 * p1 + z2 * x4 * q1 - z2 * y4 * p1 - x1 * y3 * r4 + x1 * z3 * q4
        + x1 * y4 * r3 - x1 * z4 * q3 + y1 * x3 * r4 - y1 * z3 * p4 - y1 * x4 * r3
        + y1 * z4 * p3 - z1 * x3 * q4 + z1 * y3 * p4 + z1 * x4 * q3 - z1 * y4 * p3
        - x3 * y4 * r1 + x3 * z4 * q1 + y3 * x4 * r1 - y3 * z4 * p1 - z3 * x4 * q1
        + z3 * y4 * p1 + x2 * y3 * r4 - x2 * z3 * q4 - x2 * y4 * r3 + x2 * z4 * q3
        - y2 * x3 * r4 + y2 * z3 * p4 + y2 * x4 * r3 - y2 * z4 * p3 + z2 * x3 * q4
        - z2 * y3 * p4 - z2 * x4 * q3 + z2 * y4 * p3 + x3 * y4 * r2 - x3 * z4 * q2
        - y3 * x4 * r2 + y3 * z4 * p2 + z3 * x4 * q2 - z3 * y4 * p2;
    d = (ratio)
        * (x1 * z2 * y3 - x1 * y2 * z3 + y1 * x2 * z3 - y1 * z2 * x3 - z1 * x2 * y3
           + z1 * y2 * x3 + x1 * y2 * z4 - x1 * z2 * y4 - y1 * x2 * z4 + y1 * z2 * x4
           + z1 * x2 * y4 - z1 * y2 * x4 - x1 * y3 * z4 + x1 * z3 * y4 + y1 * x3 * z4
           - y1 * z3 * x4 - z1 * x3 * y4 + z1 * y3 * x4 + x2 * y3 * z4 - x2 * z3 * y4
           - y2 * x3 * z4 + y2 * z3 * x4 + z2 * x3 * y4 - z2 * y3 * x4);


    //printf("a b c d:   %f  %f  %f  %f     %f     %f,    id0, id1, id2, id3:  %d  %d  %d  %d\n", a, b, c, d, ratio, errorRate, v0, v1, v2, v3);
    if(abs(a) <= errorRate /** errorRate*/)
    {
        if(abs(b) <= errorRate /** errorRate*/)
        {
            if(false && abs(c) <= errorRate)
            {
                t = 1;
            }
            else
            {
                t = -d / c;
            }
        }
        else
        {
            double desc = c * c - 4 * b * d;
            if(desc > 0)
            {
                t = (-c - sqrt(desc)) / (2 * b);
                if(t < 0)
                    t = (-c + sqrt(desc)) / (2 * b);
            }
            else
                t = 1;
        }
    }
    else
    {
        double results[3];
        int number = 0;
        __GEIGEN__::__NewtonSolverForCubicEquation(a, b, c, d, results, number, errorRate);

        t = 1;
        for (int index = 0;index < number;index++) {
            if (results[index] > 0 && results[index] < t) {
                t = results[index];
            }
        }
        //zs::complex<double> i(0, 1);
        //zs::complex<double> delta0(b * b - 3 * a * c, 0);
        //zs::complex<double> delta1(2 * b * b * b - 9 * a * b * c + 27 * a * a * d, 0);
        //zs::complex<double> C =
        //    pow((delta1 + sqrt(delta1 * delta1 - 4.0 * delta0 * delta0 * delta0)) / 2.0,
        //        1.0 / 3.0);
        //if(abs(C) == 0.0)
        //{
        //    // a corner case listed by wikipedia found by our collaborate from another project
        //    C = pow((delta1 - sqrt(delta1 * delta1 - 4.0 * delta0 * delta0 * delta0)) / 2.0,
        //            1.0 / 3.0);
        //}

        //zs::complex<double> u2 = (-1.0 + sqrt(3.0) * i) / 2.0;
        //zs::complex<double> u3 = (-1.0 - sqrt(3.0) * i) / 2.0;

        //zs::complex<double> t1 = (b + C + delta0 / C) / (-3.0 * a);
        //zs::complex<double> t2 = (b + u2 * C + delta0 / (u2 * C)) / (-3.0 * a);
        //zs::complex<double> t3 = (b + u3 * C + delta0 / (u3 * C)) / (-3.0 * a);
        //t                      = -1;
        //if((abs(imag(t1)) < errorRate /** errorRate*/) && (real(t1) > 0))
        //    t = real(t1);
        //if((abs(imag(t2)) < errorRate /** errorRate*/) && (real(t2) > 0)
        //   && ((real(t2) < t) || (t < 0)))
        //    t = real(t2);
        //if((abs(imag(t3)) < errorRate /** errorRate*/) && (real(t3) > 0)
        //   && ((real(t3) < t) || (t < 0)))
        //    t = real(t3);
    }
    if(t <= 0)
        t = 1;
    return t;
}

__device__ double __cal_Friction_gd_energy(const double3* _vertexes,
                                           const double3* _o_vertexes,
                                           const double3* _normal,
                                           uint32_t       gidx,
                                           double         dt,
                                           double         lastH,
                                           double         eps)
{

    double3 normal = *_normal;
    double3 Vdiff  = __GEIGEN__::__minus(_vertexes[gidx], _o_vertexes[gidx]);
    double3 VProj  = __GEIGEN__::__minus(
        Vdiff, __GEIGEN__::__s_vec_multiply(normal, __GEIGEN__::__v_vec_dot(Vdiff, normal)));
    double VProjMag2 = __GEIGEN__::__squaredNorm(VProj);
    if(VProjMag2 > eps * eps)
    {
        return lastH * (sqrt(VProjMag2) - eps * 0.5);
    }
    else
    {
        return lastH * VProjMag2 / eps * 0.5;
    }
}


__device__ double __cal_Friction_energy(const double3*         _vertexes,
                                        const double3*         _o_vertexes,
                                        int4                   MMCVIDI,
                                        double                 dt,
                                        double2                distCoord,
                                        __GEIGEN__::Matrix3x2d tanBasis,
                                        double                 lastH,
                                        double                 fricDHat,
                                        double                 eps)
{
    double3 relDX3D;
    if(MMCVIDI.x >= 0)
    {
        if(MMCVIDI.w >= 0)
        {
            Friction::computeRelDX_EE(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]),
                distCoord.x,
                distCoord.y,
                relDX3D);
        }
    }
    else
    {
        int v0I = -MMCVIDI.x - 1;
        if(MMCVIDI.z < 0)
        {
            if(MMCVIDI.y >= 0)
            {
                Friction::computeRelDX_PP(
                    __GEIGEN__::__minus(_vertexes[v0I], _o_vertexes[v0I]),
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.y],
                                        _o_vertexes[MMCVIDI.y]),
                    relDX3D);
            }
        }
        else if(MMCVIDI.w < 0)
        {
            if(MMCVIDI.y >= 0)
            {
                Friction::computeRelDX_PE(
                    __GEIGEN__::__minus(_vertexes[v0I], _o_vertexes[v0I]),
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.y],
                                        _o_vertexes[MMCVIDI.y]),
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.z],
                                        _o_vertexes[MMCVIDI.z]),
                    distCoord.x,
                    relDX3D);
            }
        }
        else
        {
            Friction::computeRelDX_PT(
                __GEIGEN__::__minus(_vertexes[v0I], _o_vertexes[v0I]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]),
                distCoord.x,
                distCoord.y,
                relDX3D);
        }
    }
    __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis);
    double                 relDXSqNorm =
        __GEIGEN__::__squaredNorm(__GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D));
    if(relDXSqNorm > fricDHat)
    {
        return lastH * sqrt(relDXSqNorm);
    }
    else
    {
        double f0;
        Friction::f0_SF(relDXSqNorm, eps, f0);
        return lastH * f0;
    }
}

__global__ void _calFrictionHessian_gd(const double3*  _vertexes,
                                       const double3*  _o_vertexes,
                                       const double3*  _normal,
                                       const uint32_t* _last_collisionPair_gd,
                                       __GEIGEN__::Matrix3x3d* H3x3,
                                       uint32_t*               D1Index,
                                       int                     number,
                                       double                  dt,
                                       double                  eps2,
                                       double*                 lastH,
                                       double                  coef)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double                 eps           = sqrt(eps2);
    int                    gidx          = _last_collisionPair_gd[idx];
    double                 multiplier_vI = coef * lastH[idx];
    __GEIGEN__::Matrix3x3d H_vI;

    double3 Vdiff  = __GEIGEN__::__minus(_vertexes[gidx], _o_vertexes[gidx]);
    double3 normal = *_normal;
    double3 VProj  = __GEIGEN__::__minus(
        Vdiff, __GEIGEN__::__s_vec_multiply(normal, __GEIGEN__::__v_vec_dot(Vdiff, normal)));
    double VProjMag2 = __GEIGEN__::__squaredNorm(VProj);

    if(VProjMag2 > eps2)
    {
        double VProjMag = sqrt(VProjMag2);

        __GEIGEN__::Matrix2x2d projH;
        __GEIGEN__::__set_Mat2x2_val_column(projH, make_double2(0, 0), make_double2(0, 0));

        double  eigenValues[2];
        int     eigenNum = 0;
        double2 eigenVecs[2];
        __GEIGEN__::__makePD2x2(VProj.x * VProj.x * -multiplier_vI / VProjMag2 / VProjMag
                                    + (multiplier_vI / VProjMag),
                                VProj.x * VProj.z * -multiplier_vI / VProjMag2 / VProjMag,
                                VProj.x * VProj.z * -multiplier_vI / VProjMag2 / VProjMag,
                                VProj.z * VProj.z * -multiplier_vI / VProjMag2 / VProjMag
                                    + (multiplier_vI / VProjMag),
                                eigenValues,
                                eigenNum,
                                eigenVecs);
        for(int i = 0; i < eigenNum; i++)
        {
            if(eigenValues[i] > 0)
            {
                __GEIGEN__::Matrix2x2d eigenMatrix =
                    __GEIGEN__::__v2_vec2_toMat2x2(eigenVecs[i], eigenVecs[i]);
                eigenMatrix =
                    __GEIGEN__::__s_Mat2x2_multiply(eigenMatrix, eigenValues[i]);
                projH = __GEIGEN__::__Mat2x2_add(projH, eigenMatrix);
            }
        }

        __GEIGEN__::__set_Mat_val(H_vI,
                                  projH.m[0][0],
                                  0,
                                  projH.m[0][1],
                                  0,
                                  0,
                                  0,
                                  projH.m[1][0],
                                  0,
                                  projH.m[1][1]);
    }
    else
    {
        __GEIGEN__::__set_Mat_val(
            H_vI, (multiplier_vI / eps), 0, 0, 0, 0, 0, 0, 0, (multiplier_vI / eps));
    }

    H3x3[idx]    = H_vI;
    D1Index[idx] = gidx;
}

__global__ void _calFrictionHessian(const double3* _vertexes,
                                    const double3* _o_vertexes,
                                    const int4*    _last_collisionPair,
                                    __GEIGEN__::Matrix12x12d* H12x12,
                                    __GEIGEN__::Matrix9x9d*   H9x9,
                                    __GEIGEN__::Matrix6x6d*   H6x6,
                                    uint4*                    D4Index,
                                    uint3*                    D3Index,
                                    uint2*                    D2Index,
                                    uint32_t*                 _cpNum,
                                    int                       number,
                                    double                    dt,
                                    double2*                  distCoord,
                                    __GEIGEN__::Matrix3x2d*   tanBasis,
                                    double                    eps2,
                                    double*                   lastH,
                                    double                    coef,
                                    int                       offset4,
                                    int                       offset3,
                                    int                       offset2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4    MMCVIDI = _last_collisionPair[idx];
    double  eps     = sqrt(eps2);
    double3 relDX3D;
    if(MMCVIDI.x >= 0)
    {
        Friction::computeRelDX_EE(
            __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
            __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
            __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]),
            distCoord[idx].x,
            distCoord[idx].y,
            relDX3D);


        __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
        double2 relDX       = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);
        double  relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
        double  relDXNorm   = sqrt(relDXSqNorm);
        __GEIGEN__::Matrix12x2d T;
        Friction::computeT_EE(tanBasis[idx], distCoord[idx].x, distCoord[idx].y, T);
        __GEIGEN__::Matrix2x2d M2;
        if(relDXSqNorm > eps2)
        {
            __GEIGEN__::__set_Mat_identity(M2);
            M2.m[0][0] /= relDXNorm;
            M2.m[1][1] /= relDXNorm;
            M2 = __GEIGEN__::__Mat2x2_minus(
                M2,
                __GEIGEN__::__s_Mat2x2_multiply(__GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                                                1 / (relDXSqNorm * relDXNorm)));
        }
        else
        {
            double f1_div_relDXNorm;
            Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
            double f2;
            Friction::f2_SF(relDXSqNorm, eps, f2);
            if(f2 != f1_div_relDXNorm && relDXSqNorm)
            {

                __GEIGEN__::__set_Mat_identity(M2);
                M2.m[0][0] *= f1_div_relDXNorm;
                M2.m[1][1] *= f1_div_relDXNorm;
                M2 = __GEIGEN__::__Mat2x2_minus(
                    M2,
                    __GEIGEN__::__s_Mat2x2_multiply(__GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                                                    (f1_div_relDXNorm - f2) / relDXSqNorm));
            }
            else
            {
                __GEIGEN__::__set_Mat_identity(M2);
                M2.m[0][0] *= f1_div_relDXNorm;
                M2.m[1][1] *= f1_div_relDXNorm;
            }
        }

        __GEIGEN__::Matrix2x2d projH;
        __GEIGEN__::__set_Mat2x2_val_column(projH, make_double2(0, 0), make_double2(0, 0));

        double  eigenValues[2];
        int     eigenNum = 0;
        double2 eigenVecs[2];
        __GEIGEN__::__makePD2x2(
            M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1], eigenValues, eigenNum, eigenVecs);
        for(int i = 0; i < eigenNum; i++)
        {
            if(eigenValues[i] > 0)
            {
                __GEIGEN__::Matrix2x2d eigenMatrix =
                    __GEIGEN__::__v2_vec2_toMat2x2(eigenVecs[i], eigenVecs[i]);
                eigenMatrix =
                    __GEIGEN__::__s_Mat2x2_multiply(eigenMatrix, eigenValues[i]);
                projH = __GEIGEN__::__Mat2x2_add(projH, eigenMatrix);
            }
        }

        __GEIGEN__::Matrix12x2d TM2 = __GEIGEN__::__M12x2_M2x2_Multiply(T, projH);

        __GEIGEN__::Matrix12x12d HessianBlock =
            __GEIGEN__::__s_M12x12_Multiply(__M12x2_M12x2T_Multiply(TM2, T),
                                            coef * lastH[idx]);
        int Hidx = gipc::ATOMIC_ADD(_cpNum + 4, 1);
        Hidx += offset4;
        H12x12[Hidx]  = HessianBlock;
        D4Index[Hidx] = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
    }
    else
    {
        int v0I = -MMCVIDI.x - 1;
        if(MMCVIDI.z < 0)
        {

            MMCVIDI.x = v0I;
            Friction::computeRelDX_PP(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                relDX3D);

            __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
            double2 relDX       = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);
            double  relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
            double  relDXNorm   = sqrt(relDXSqNorm);
            __GEIGEN__::Matrix6x2d T;
            Friction::computeT_PP(tanBasis[idx], T);
            __GEIGEN__::Matrix2x2d M2;
            if(relDXSqNorm > eps2)
            {
                __GEIGEN__::__set_Mat_identity(M2);
                M2.m[0][0] /= relDXNorm;
                M2.m[1][1] /= relDXNorm;
                M2 = __GEIGEN__::__Mat2x2_minus(
                    M2,
                    __GEIGEN__::__s_Mat2x2_multiply(__GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                                                    1 / (relDXSqNorm * relDXNorm)));
            }
            else
            {
                double f1_div_relDXNorm;
                Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                double f2;
                Friction::f2_SF(relDXSqNorm, eps, f2);
                if(f2 != f1_div_relDXNorm && relDXSqNorm)
                {

                    __GEIGEN__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                    M2 = __GEIGEN__::__Mat2x2_minus(
                        M2,
                        __GEIGEN__::__s_Mat2x2_multiply(
                            __GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                            (f1_div_relDXNorm - f2) / relDXSqNorm));
                }
                else
                {
                    __GEIGEN__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                }
            }
            __GEIGEN__::Matrix2x2d projH;
            __GEIGEN__::__set_Mat2x2_val_column(projH, make_double2(0, 0), make_double2(0, 0));

            double  eigenValues[2];
            int     eigenNum = 0;
            double2 eigenVecs[2];
            __GEIGEN__::__makePD2x2(
                M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1], eigenValues, eigenNum, eigenVecs);
            for(int i = 0; i < eigenNum; i++)
            {
                if(eigenValues[i] > 0)
                {
                    __GEIGEN__::Matrix2x2d eigenMatrix =
                        __GEIGEN__::__v2_vec2_toMat2x2(eigenVecs[i], eigenVecs[i]);
                    eigenMatrix =
                        __GEIGEN__::__s_Mat2x2_multiply(eigenMatrix, eigenValues[i]);
                    projH = __GEIGEN__::__Mat2x2_add(projH, eigenMatrix);
                }
            }

            __GEIGEN__::Matrix6x2d TM2 = __GEIGEN__::__M6x2_M2x2_Multiply(T, projH);

            __GEIGEN__::Matrix6x6d HessianBlock =
                __GEIGEN__::__s_M6x6_Multiply(__M6x2_M6x2T_Multiply(TM2, T),
                                              coef * lastH[idx]);

            int Hidx = gipc::ATOMIC_ADD(_cpNum + 2, 1);
            Hidx += offset2;
            H6x6[Hidx]    = HessianBlock;
            D2Index[Hidx] = make_uint2(MMCVIDI.x, MMCVIDI.y);
        }
        else if(MMCVIDI.w < 0)
        {

            MMCVIDI.x = v0I;
            Friction::computeRelDX_PE(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                distCoord[idx].x,
                relDX3D);

            __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
            double2 relDX       = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);
            double  relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
            double  relDXNorm   = sqrt(relDXSqNorm);
            __GEIGEN__::Matrix9x2d T;
            Friction::computeT_PE(tanBasis[idx], distCoord[idx].x, T);
            __GEIGEN__::Matrix2x2d M2;
            if(relDXSqNorm > eps2)
            {
                __GEIGEN__::__set_Mat_identity(M2);
                M2.m[0][0] /= relDXNorm;
                M2.m[1][1] /= relDXNorm;
                M2 = __GEIGEN__::__Mat2x2_minus(
                    M2,
                    __GEIGEN__::__s_Mat2x2_multiply(__GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                                                    1 / (relDXSqNorm * relDXNorm)));
            }
            else
            {
                double f1_div_relDXNorm;
                Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                double f2;
                Friction::f2_SF(relDXSqNorm, eps, f2);
                if(f2 != f1_div_relDXNorm && relDXSqNorm)
                {

                    __GEIGEN__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                    M2 = __GEIGEN__::__Mat2x2_minus(
                        M2,
                        __GEIGEN__::__s_Mat2x2_multiply(
                            __GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                            (f1_div_relDXNorm - f2) / relDXSqNorm));
                }
                else
                {
                    __GEIGEN__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                }
            }
            __GEIGEN__::Matrix2x2d projH;
            __GEIGEN__::__set_Mat2x2_val_column(projH, make_double2(0, 0), make_double2(0, 0));

            double  eigenValues[2];
            int     eigenNum = 0;
            double2 eigenVecs[2];
            __GEIGEN__::__makePD2x2(
                M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1], eigenValues, eigenNum, eigenVecs);
            for(int i = 0; i < eigenNum; i++)
            {
                if(eigenValues[i] > 0)
                {
                    __GEIGEN__::Matrix2x2d eigenMatrix =
                        __GEIGEN__::__v2_vec2_toMat2x2(eigenVecs[i], eigenVecs[i]);
                    eigenMatrix =
                        __GEIGEN__::__s_Mat2x2_multiply(eigenMatrix, eigenValues[i]);
                    projH = __GEIGEN__::__Mat2x2_add(projH, eigenMatrix);
                }
            }

            __GEIGEN__::Matrix9x2d TM2 = __GEIGEN__::__M9x2_M2x2_Multiply(T, projH);

            __GEIGEN__::Matrix9x9d HessianBlock =
                __GEIGEN__::__s_M9x9_Multiply(__M9x2_M9x2T_Multiply(TM2, T),
                                              coef * lastH[idx]);
            int Hidx = gipc::ATOMIC_ADD(_cpNum + 3, 1);
            Hidx += offset3;
            H9x9[Hidx]    = HessianBlock;
            D3Index[Hidx] = make_uint3(v0I, MMCVIDI.y, MMCVIDI.z);
        }
        else
        {
            MMCVIDI.x = v0I;
            Friction::computeRelDX_PT(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]),
                distCoord[idx].x,
                distCoord[idx].y,
                relDX3D);


            __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
            double2 relDX       = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);
            double  relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
            double  relDXNorm   = sqrt(relDXSqNorm);
            __GEIGEN__::Matrix12x2d T;
            Friction::computeT_PT(
                tanBasis[idx], distCoord[idx].x, distCoord[idx].y, T);
            __GEIGEN__::Matrix2x2d M2;
            if(relDXSqNorm > eps2)
            {
                __GEIGEN__::__set_Mat_identity(M2);
                M2.m[0][0] /= relDXNorm;
                M2.m[1][1] /= relDXNorm;
                M2 = __GEIGEN__::__Mat2x2_minus(
                    M2,
                    __GEIGEN__::__s_Mat2x2_multiply(__GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                                                    1 / (relDXSqNorm * relDXNorm)));
            }
            else
            {
                double f1_div_relDXNorm;
                Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                double f2;
                Friction::f2_SF(relDXSqNorm, eps, f2);
                if(f2 != f1_div_relDXNorm && relDXSqNorm)
                {

                    __GEIGEN__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                    M2 = __GEIGEN__::__Mat2x2_minus(
                        M2,
                        __GEIGEN__::__s_Mat2x2_multiply(
                            __GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                            (f1_div_relDXNorm - f2) / relDXSqNorm));
                }
                else
                {
                    __GEIGEN__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                }
            }
            __GEIGEN__::Matrix2x2d projH;
            __GEIGEN__::__set_Mat2x2_val_column(projH, make_double2(0, 0), make_double2(0, 0));

            double  eigenValues[2];
            int     eigenNum = 0;
            double2 eigenVecs[2];
            __GEIGEN__::__makePD2x2(
                M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1], eigenValues, eigenNum, eigenVecs);
            for(int i = 0; i < eigenNum; i++)
            {
                if(eigenValues[i] > 0)
                {
                    __GEIGEN__::Matrix2x2d eigenMatrix =
                        __GEIGEN__::__v2_vec2_toMat2x2(eigenVecs[i], eigenVecs[i]);
                    eigenMatrix =
                        __GEIGEN__::__s_Mat2x2_multiply(eigenMatrix, eigenValues[i]);
                    projH = __GEIGEN__::__Mat2x2_add(projH, eigenMatrix);
                }
            }

            __GEIGEN__::Matrix12x2d TM2 = __GEIGEN__::__M12x2_M2x2_Multiply(T, projH);

            __GEIGEN__::Matrix12x12d HessianBlock =
                __GEIGEN__::__s_M12x12_Multiply(__M12x2_M12x2T_Multiply(TM2, T),
                                                coef * lastH[idx]);
            int Hidx = gipc::ATOMIC_ADD(_cpNum + 4, 1);
            Hidx += offset4;
            H12x12[Hidx]  = HessianBlock;
            D4Index[Hidx] = make_uint4(v0I, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
        }
    }
}

__global__ void _calBarrierGradientAndHessian(const double3*    _vertexes,
                                              const double3*    _rest_vertexes,
                                              const int4* _collisionPair,
                                              double3*          _gradient,
                                              __GEIGEN__::Matrix12x12d* H12x12,
                                              __GEIGEN__::Matrix9x9d*   H9x9,
                                              __GEIGEN__::Matrix6x6d*   H6x6,
                                              uint4*                    D4Index,
                                              uint3*                    D3Index,
                                              uint2*                    D2Index,
                                              uint32_t*                 _cpNum,
                                              int*   matIndex,
                                              double dHat,
                                              double Kappa,
                                              int    number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4   MMCVIDI   = _collisionPair[idx];
    double dHat_sqrt = sqrt(dHat);
    //double dHat = dHat_sqrt * dHat_sqrt;
    //double Kappa = 1;
    double gassThreshold = 1e-4;
    if(MMCVIDI.x >= 0)
    {
        if(MMCVIDI.w >= 0)
        {
#ifdef NEWF
            double dis;
            _d_EE(_vertexes[MMCVIDI.x],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            dis                                = sqrt(dis);
            double                  d_hat_sqrt = sqrt(dHat);
            __GEIGEN__::Matrix12x9d PFPxT;
            pFpx_ee2(_vertexes[MMCVIDI.x],
                     _vertexes[MMCVIDI.y],
                     _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w],
                     d_hat_sqrt,
                     PFPxT);
            double              I5 = pow(dis / d_hat_sqrt, 2);
            __GEIGEN__::Vector9 tmp;
            tmp.v[0] = tmp.v[1] = tmp.v[2] = tmp.v[3] = tmp.v[4] = tmp.v[5] =
                tmp.v[6] = tmp.v[7] = 0;
            tmp.v[8]                = dis / d_hat_sqrt;

            // Rank-1 factorization: q0 == e8, so H = lambda0 * e8 * e8^T has a
            // single nonzero entry and Hessian = PFPxT * H * PFPxT^T collapses
            // to lambda0 * c8 * c8^T with c8 = column 8 of PFPxT.  No 9x9/12x12
            // local matrix is materialized (formerly a 2128B stack frame).
#else

            double3 v0 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
            double3 v1 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.x]);
            double3 v2 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
            __GEIGEN__::Matrix3x3d Ds;
            __GEIGEN__::__set_Mat_val_column(Ds, v0, v1, v2);
            double3 normal = __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(
                v0, __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z])));
            double  dis    = __GEIGEN__::__v_vec_dot(v1, normal);
            if(dis < 0)
            {
                normal = make_double3(-normal.x, -normal.y, -normal.z);
                dis    = -dis;
            }

            double3 pos2 =
                __GEIGEN__::__add(_vertexes[MMCVIDI.z],
                                  __GEIGEN__::__s_vec_multiply(normal, dHat_sqrt - dis));
            double3 pos3 =
                __GEIGEN__::__add(_vertexes[MMCVIDI.w],
                                  __GEIGEN__::__s_vec_multiply(normal, dHat_sqrt - dis));

            double3 u0 = v0;
            double3 u1 = __GEIGEN__::__minus(pos2, _vertexes[MMCVIDI.x]);
            double3 u2 = __GEIGEN__::__minus(pos3, _vertexes[MMCVIDI.x]);

            __GEIGEN__::Matrix3x3d Dm, DmInv;
            __GEIGEN__::__set_Mat_val_column(Dm, u0, u1, u2);

            __GEIGEN__::__Inverse(Dm, DmInv);

            __GEIGEN__::Matrix3x3d F;
            __GEIGEN__::__M_Mat_multiply(Ds, DmInv, F);

            double3 FxN = __GEIGEN__::__M_v_multiply(F, normal);
            double  I5  = __GEIGEN__::__squaredNorm(FxN);

            __GEIGEN__::Matrix9x12d PFPx = __computePFDsPX3D_double(DmInv);

            __GEIGEN__::Matrix3x3d fnn;

            __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(normal, normal);

            __GEIGEN__::__M_Mat_multiply(F, nn, fnn);

            __GEIGEN__::Vector9 tmp = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);

#endif

#if(RANK == 1)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif(RANK == 2)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);
#elif(RANK == 3)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                       * (3 * I5 + 2 * I5 * log(I5) - 3))
                    / I5);
#elif(RANK == 4)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                 * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                    / I5);
#elif(RANK == 5)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                    / I5);
#elif(RANK == 6)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                 * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                    / I5);
#endif


#if(RANK == 1)
            double lambda0 =
                Kappa
                * (2 * dHat * dHat
                   * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1))
                / I5;
            if(dis * dis < gassThreshold * dHat)
            {
                double lambda1 =
                    Kappa
                    * (2 * dHat * dHat
                       * (6 * gassThreshold + 2 * gassThreshold * log(gassThreshold)
                          - 7 * gassThreshold * gassThreshold
                          - 6 * gassThreshold * gassThreshold * log(gassThreshold) + 1))
                    / gassThreshold;
                lambda0 = lambda1;
            }
#elif(RANK == 2)
            double lambda0 =
                -(4 * Kappa * dHat * dHat
                  * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 6 * I5 * log(I5)
                     - 2 * I5 * I5 + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                / I5;
            if(dis * dis < gassThreshold * dHat)
            {
                double lambda1 =
                    -(4 * Kappa * dHat * dHat
                      * (4 * gassThreshold + log(gassThreshold)
                         - 3 * gassThreshold * gassThreshold * log(gassThreshold) * log(gassThreshold)
                         + 6 * gassThreshold * log(gassThreshold) - 2 * gassThreshold * gassThreshold
                         + gassThreshold * log(gassThreshold) * log(gassThreshold)
                         - 7 * gassThreshold * gassThreshold * log(gassThreshold) - 2))
                    / gassThreshold;
                lambda0 = lambda1;
            }
#elif(RANK == 3)
            double lambda0 =
                (2 * Kappa * dHat * dHat * log(I5)
                 * (24 * I5 + 3 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                    + 18 * I5 * log(I5) - 12 * I5 * I5
                    + 2 * I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 12))
                / I5;
#elif(RANK == 4)
            double lambda0 =
                -(4 * Kappa * dHat * dHat * log(I5) * log(I5)
                  * (24 * I5 + 2 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                     + 12 * I5 * log(I5) - 12 * I5 * I5 + I5 * log(I5) * log(I5)
                     - 14 * I5 * I5 * log(I5) - 12))
                / I5;
#elif(RANK == 5)
            double lambda0 =
                (2 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                 * (80 * I5 + 5 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                    + 30 * I5 * log(I5) - 40 * I5 * I5
                    + 2 * I5 * log(I5) * log(I5) - 35 * I5 * I5 * log(I5) - 40))
                / I5;
#elif(RANK == 6)
            double lambda0 =
                -(4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                  * (60 * I5 + 3 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                     + 18 * I5 * log(I5) - 30 * I5 * I5 + I5 * log(I5) * log(I5)
                     - 21 * I5 * I5 * log(I5) - 30))
                / I5;
#endif


#ifdef NEWF
            // gradient_vec = PFPxT * flatten_pk1, and flatten_pk1 has only
            // v[8] nonzero, so gradient_vec = flatten_pk1.v[8] * c8.
            __GEIGEN__::Vector12 c8;
#pragma unroll
            for(int i = 0; i < 12; i++)
                c8.v[i] = PFPxT.m[i][8];

            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__s_vec12_multiply(c8, flatten_pk1.v[8]);
#else

            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply(__GEIGEN__::__Transpose9x12(PFPx), flatten_pk1);
            //__GEIGEN__::Matrix3x3d Q0;

            //            __GEIGEN__::Matrix3x3d fnn;

            //           __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(normal, normal);

            //            __GEIGEN__::__M_Mat_multiply(F, nn, fnn);

            __GEIGEN__::Vector9 q0 = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);

            q0 = __GEIGEN__::__s_vec9_multiply(q0, 1.0 / sqrt(I5));

            __GEIGEN__::Matrix9x9d H;
            __GEIGEN__::__init_Mat9x9(H, 0);

            H = __GEIGEN__::__S_Mat9x9_multiply(__GEIGEN__::__v9_vec9_toMat9x9(q0, q0), lambda0);

            __GEIGEN__::Matrix12x9d PFPxTransPos = __GEIGEN__::__Transpose9x12(PFPx);
            __GEIGEN__::Matrix12x12d Hessian = __GEIGEN__::__M12x9_M9x12_Multiply(
                __GEIGEN__::__M12x9_M9x9_Multiply(PFPxTransPos, H), PFPx);
#endif

            {
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
            }
            int Hidx = matIndex[idx];  //gipc::ATOMIC_ADD(_cpNum + 4, 1);

#ifdef NEWF
            __storeHessian12Rank1(H12x12, Hidx, lambda0, c8);
#else
            H12x12[Hidx] = Hessian;
#endif
            D4Index[Hidx] = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
        }
        else
        {
            //return;
            MMCVIDI.w = -MMCVIDI.w - 1;
            double3 v0 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
            double3 v1 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z]);
            double c  = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1)) /*/ __GEIGEN__::__norm(v0)*/;
            double I1 = c * c;
            if(I1 == 0)
                return;
            double dis;
            _d_EE(_vertexes[MMCVIDI.x],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            double I2 = dis / dHat;
            dis       = sqrt(dis);

            __GEIGEN__::Matrix3x3d F;
            __GEIGEN__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
            double3 n1 = make_double3(0, 1, 0);
            double3 n2 = make_double3(0, 0, 1);

            double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                        _rest_vertexes[MMCVIDI.y],
                                        _rest_vertexes[MMCVIDI.z],
                                        _rest_vertexes[MMCVIDI.w]);

            __GEIGEN__::Matrix3x3d g1, g2;

            __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(n1, n1);
            __GEIGEN__::__M_Mat_multiply(F, nn, g1);
            nn = __GEIGEN__::__v_vec_toMat(n2, n2);
            __GEIGEN__::__M_Mat_multiply(F, nn, g2);

            __GEIGEN__::Vector9 flatten_g1 = __GEIGEN__::__Mat3x3_to_vec9_double(g1);
            __GEIGEN__::Vector9 flatten_g2 = __GEIGEN__::__Mat3x3_to_vec9_double(g2);

            __GEIGEN__::Matrix12x9d PFPx;
            pFpx_pee(_vertexes[MMCVIDI.x],
                     _vertexes[MMCVIDI.y],
                     _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w],
                     dHat_sqrt,
                     PFPx);

#if(RANK == 1)
            double p1 = Kappa * 2
                        * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                        / (eps_x * eps_x);
            double p2 = Kappa * 2
                        * (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1)
                           * (I2 + 2 * I2 * log(I2) - 1))
                        / (I2 * eps_x * eps_x);
#elif(RANK == 2)
            double p1 = -Kappa * 2
                        * (2 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x)
                           * (I2 - 1) * (I2 - 1))
                        / (eps_x * eps_x);
            double p2 = -Kappa * 2
                        * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x)
                           * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                        / (I2 * (eps_x * eps_x));
#elif(RANK == 4)
            double p1 = -Kappa * 2
                        * (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x)
                           * (I2 - 1) * (I2 - 1))
                        / (eps_x * eps_x);
            double p2 = -Kappa * 2
                        * (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x)
                           * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                        / (I2 * (eps_x * eps_x));
#elif(RANK == 6)
            double p1 = -Kappa * 2
                        * (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x)
                           * (I2 - 1) * (I2 - 1))
                        / (eps_x * eps_x);
            double p2 = -Kappa * 2
                        * (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x)
                           * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                        / (I2 * (eps_x * eps_x));
#endif
            __GEIGEN__::Vector9 flatten_pk1 =
                __GEIGEN__::__add9(__GEIGEN__::__s_vec9_multiply(flatten_g1, p1),
                                   __GEIGEN__::__s_vec9_multiply(flatten_g2, p2));
            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply(PFPx, flatten_pk1);

            {
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
            }

#if(RANK == 1)
            double lambda10 =
                Kappa * (4 * dHat * dHat * log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                / (eps_x * eps_x);
            double lambda11 =
                Kappa * 2
                * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                / (eps_x * eps_x);
            double lambda12 =
                Kappa * 2
                * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                / (eps_x * eps_x);
#elif(RANK == 2)
            double lambda10 = -Kappa
                              * (4 * dHat * dHat * log(I2) * log(I2) * (I2 - 1)
                                 * (I2 - 1) * (3 * I1 - eps_x))
                              / (eps_x * eps_x);
            double lambda11 = -Kappa
                              * (4 * dHat * dHat * log(I2) * log(I2)
                                 * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                              / (eps_x * eps_x);
            double lambda12 = -Kappa
                              * (4 * dHat * dHat * log(I2) * log(I2)
                                 * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                              / (eps_x * eps_x);
#elif(RANK == 4)
            double lambda10 = -Kappa
                              * (4 * dHat * dHat * pow(log(I2), 4) * (I2 - 1)
                                 * (I2 - 1) * (3 * I1 - eps_x))
                              / (eps_x * eps_x);
            double lambda11 = -Kappa
                              * (4 * dHat * dHat * pow(log(I2), 4)
                                 * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                              / (eps_x * eps_x);
            double lambda12 = -Kappa
                              * (4 * dHat * dHat * pow(log(I2), 4)
                                 * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                              / (eps_x * eps_x);
#elif(RANK == 6)
            double lambda10 = -Kappa
                              * (4 * dHat * dHat * pow(log(I2), 6) * (I2 - 1)
                                 * (I2 - 1) * (3 * I1 - eps_x))
                              / (eps_x * eps_x);
            double lambda11 = -Kappa
                              * (4 * dHat * dHat * pow(log(I2), 6)
                                 * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                              / (eps_x * eps_x);
            double lambda12 = -Kappa
                              * (4 * dHat * dHat * pow(log(I2), 6)
                                 * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                              / (eps_x * eps_x);
#endif
            __GEIGEN__::Matrix3x3d Tx, Ty, Tz;
            __GEIGEN__::__set_Mat_val(Tx, 0, 0, 0, 0, 0, 1, 0, -1, 0);
            __GEIGEN__::__set_Mat_val(Ty, 0, 0, -1, 0, 0, 0, 1, 0, 0);
            __GEIGEN__::__set_Mat_val(Tz, 0, 1, 0, -1, 0, 0, 0, 0, 0);

            __GEIGEN__::Vector9 q11 = __GEIGEN__::__Mat3x3_to_vec9_double(
                __GEIGEN__::__M_Mat_multiply(Tx, g1));
            __GEIGEN__::__normalized_vec9_double(q11);
            __GEIGEN__::Vector9 q12 = __GEIGEN__::__Mat3x3_to_vec9_double(
                __GEIGEN__::__M_Mat_multiply(Tz, g1));
            __GEIGEN__::__normalized_vec9_double(q12);

            // projectedH = lambda11*q11*q11^T + lambda12*q12*q12^T + (eigen terms)
            // is never materialized: Hessian = PFPx*projectedH*PFPx^T
            //          = sum_k lambdas[k]*w_k*w_k^T with w_k = PFPx*q_k.
            __GEIGEN__::Vector12 w0 = __GEIGEN__::__M12x9_v9_multiply(PFPx, q11);
            __GEIGEN__::Vector12 w1 = __GEIGEN__::__M12x9_v9_multiply(PFPx, q12);

#if(RANK == 1)
            double lambda20 =
                -Kappa
                * (2 * I1 * dHat * dHat * (I1 - 2 * eps_x)
                   * (6 * I2 + 2 * I2 * log(I2) - 7 * I2 * I2 - 6 * I2 * I2 * log(I2) + 1))
                / (I2 * eps_x * eps_x);
#elif(RANK == 2)
            double lambda20 =
                Kappa
                * (4 * I1 * dHat * dHat * (I1 - 2 * eps_x)
                   * (4 * I2 + log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                      + 6 * I2 * log(I2) - 2 * I2 * I2 + I2 * log(I2) * log(I2)
                      - 7 * I2 * I2 * log(I2) - 2))
                / (I2 * (eps_x * eps_x));
#elif(RANK == 4)
            double lambda20 =
                Kappa
                * (4 * I1 * dHat * dHat * log(I2) * log(I2) * (I1 - 2 * eps_x)
                   * (24 * I2 + 2 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                      + 12 * I2 * log(I2) - 12 * I2 * I2
                      + I2 * log(I2) * log(I2) - 14 * I2 * I2 * log(I2) - 12))
                / (I2 * (eps_x * eps_x));
#elif(RANK == 6)
            double lambda20 =
                Kappa
                * (4 * I1 * dHat * dHat * pow(log(I2), 4) * (I1 - 2 * eps_x)
                   * (60 * I2 + 3 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                      + 18 * I2 * log(I2) - 30 * I2 * I2
                      + I2 * log(I2) * log(I2) - 21 * I2 * I2 * log(I2) - 30))
                / (I2 * (eps_x * eps_x));
#endif

#if(RANK == 1)
            double lambdag1g = Kappa * 4 * c * F.m[2][2]
                               * ((2 * dHat * dHat * (I1 - eps_x) * (I2 - 1)
                                   * (I2 + 2 * I2 * log(I2) - 1))
                                  / (I2 * eps_x * eps_x));
#elif(RANK == 2)
            double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                               * (4 * dHat * dHat * log(I2) * (I1 - eps_x)
                                  * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                               / (I2 * (eps_x * eps_x));
#elif(RANK == 4)
            double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                               * (4 * dHat * dHat * pow(log(I2), 3) * (I1 - eps_x)
                                  * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                               / (I2 * (eps_x * eps_x));
#elif(RANK == 6)
            double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                               * (4 * dHat * dHat * pow(log(I2), 5) * (I1 - eps_x)
                                  * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                               / (I2 * (eps_x * eps_x));
#endif
            double  eigenValues[2];
            int     eigenNum = 0;
            double2 eigenVecs[2];
            __GEIGEN__::__makePD2x2(
                lambda10, lambdag1g, lambdag1g, lambda20, eigenValues, eigenNum, eigenVecs);

            __GEIGEN__::Vector12 w2, w3;
            double             l2 = 0, l3 = 0;
#pragma unroll
            for(int z = 0; z < 12; z++)
            {
                w2.v[z] = 0;
                w3.v[z] = 0;
            }

            for(int i = 0; i < eigenNum; i++)
            {
                if(eigenValues[i] > 0)
                {
                    __GEIGEN__::Matrix3x3d eigenMatrix;
                    __GEIGEN__::__set_Mat_val(eigenMatrix,
                                              0,
                                              0,
                                              0,
                                              0,
                                              eigenVecs[i].x,
                                              0,
                                              0,
                                              0,
                                              eigenVecs[i].y);

                    __GEIGEN__::Vector9 eigenMVec =
                        __GEIGEN__::__Mat3x3_to_vec9_double(eigenMatrix);

                    // rank component eigenValues[i] * (PFPx*eigenMVec) * (PFPx*eigenMVec)^T
                    if(i == 0)
                    {
                        l2 = eigenValues[0];
                        w2 = __GEIGEN__::__M12x9_v9_multiply(PFPx, eigenMVec);
                    }
                    else
                    {
                        l3 = eigenValues[i];
                        w3 = __GEIGEN__::__M12x9_v9_multiply(PFPx, eigenMVec);
                    }
                }
            }

            int Hidx = matIndex[idx];  //int Hidx = gipc::ATOMIC_ADD(_cpNum + 4, 1);

            __storeHessian12RankN(H12x12, Hidx, lambda11, lambda12, l2, l3, w0, w1, w2, w3);
            D4Index[Hidx] = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
        }
    }
    else
    {
        int v0I = -MMCVIDI.x - 1;
        if(MMCVIDI.z < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.z = -MMCVIDI.z - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;
                MMCVIDI.x = v0I;
                double3 v0 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.x]);
                double3 v1 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.y]);
                double c  = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1)) /*/ __GEIGEN__::__norm(v0)*/;
                double I1 = c * c;
                if(I1 == 0)
                    return;
                double dis;
                _d_PP(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], dis);
                double I2 = dis / dHat;
                dis       = sqrt(dis);

                __GEIGEN__::Matrix3x3d F;
                __GEIGEN__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
                double3 n1 = make_double3(0, 1, 0);
                double3 n2 = make_double3(0, 0, 1);

                double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                            _rest_vertexes[MMCVIDI.z],
                                            _rest_vertexes[MMCVIDI.y],
                                            _rest_vertexes[MMCVIDI.w]);

                __GEIGEN__::Matrix3x3d g1, g2;

                __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(n1, n1);
                __GEIGEN__::__M_Mat_multiply(F, nn, g1);
                nn = __GEIGEN__::__v_vec_toMat(n2, n2);
                __GEIGEN__::__M_Mat_multiply(F, nn, g2);

                __GEIGEN__::Vector9 flatten_g1 = __GEIGEN__::__Mat3x3_to_vec9_double(g1);
                __GEIGEN__::Vector9 flatten_g2 = __GEIGEN__::__Mat3x3_to_vec9_double(g2);

                __GEIGEN__::Matrix12x9d PFPx;
                pFpx_ppp(_vertexes[MMCVIDI.x],
                         _vertexes[MMCVIDI.y],
                         _vertexes[MMCVIDI.z],
                         _vertexes[MMCVIDI.w],
                         dHat_sqrt,
                         PFPx);

#if(RANK == 1)
                double p1 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
                double p2 = Kappa * 2
                            * (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1)
                               * (I2 + 2 * I2 * log(I2) - 1))
                            / (I2 * eps_x * eps_x);
#elif(RANK == 2)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * log(I2) * log(I2)
                               * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                            / (I2 * (eps_x * eps_x));
#elif(RANK == 4)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x)
                               * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                            / (I2 * (eps_x * eps_x));
#elif(RANK == 6)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x)
                               * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                            / (I2 * (eps_x * eps_x));
#endif
                __GEIGEN__::Vector9 flatten_pk1 =
                    __GEIGEN__::__add9(__GEIGEN__::__s_vec9_multiply(flatten_g1, p1),
                                       __GEIGEN__::__s_vec9_multiply(flatten_g2, p2));
                __GEIGEN__::Vector12 gradient_vec =
                    __GEIGEN__::__M12x9_v9_multiply(PFPx, flatten_pk1);

                {
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
                }

#if(RANK == 1)
                double lambda10 =
                    Kappa * (4 * dHat * dHat * log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                    / (eps_x * eps_x);
                double lambda11 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
                double lambda12 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
#elif(RANK == 2)
                double lambda10 = -Kappa
                                  * (4 * dHat * dHat * log(I2) * log(I2)
                                     * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                                  / (eps_x * eps_x);
                double lambda11 = -Kappa
                                  * (4 * dHat * dHat * log(I2) * log(I2)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
                double lambda12 = -Kappa
                                  * (4 * dHat * dHat * log(I2) * log(I2)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
#elif(RANK == 4)
                double lambda10 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 4)
                                     * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                                  / (eps_x * eps_x);
                double lambda11 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 4)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
                double lambda12 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 4)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
#elif(RANK == 6)
                double lambda10 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 6)
                                     * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                                  / (eps_x * eps_x);
                double lambda11 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 6)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
                double lambda12 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 6)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
#endif
                __GEIGEN__::Matrix3x3d Tx, Ty, Tz;
                __GEIGEN__::__set_Mat_val(Tx, 0, 0, 0, 0, 0, 1, 0, -1, 0);
                __GEIGEN__::__set_Mat_val(Ty, 0, 0, -1, 0, 0, 0, 1, 0, 0);
                __GEIGEN__::__set_Mat_val(Tz, 0, 1, 0, -1, 0, 0, 0, 0, 0);

                __GEIGEN__::Vector9 q11 = __GEIGEN__::__Mat3x3_to_vec9_double(
                    __GEIGEN__::__M_Mat_multiply(Tx, g1));
                __GEIGEN__::__normalized_vec9_double(q11);
                __GEIGEN__::Vector9 q12 = __GEIGEN__::__Mat3x3_to_vec9_double(
                    __GEIGEN__::__M_Mat_multiply(Tz, g1));
                __GEIGEN__::__normalized_vec9_double(q12);

                // projectedH is never materialized: Hessian = PFPx*projectedH*PFPx^T
                //          = sum_k lambdas[k]*w_k*w_k^T with w_k = PFPx*q_k.
                __GEIGEN__::Vector12 w0 = __GEIGEN__::__M12x9_v9_multiply(PFPx, q11);
                __GEIGEN__::Vector12 w1 = __GEIGEN__::__M12x9_v9_multiply(PFPx, q12);

#if(RANK == 1)
                double lambda20 = -Kappa
                                  * (2 * I1 * dHat * dHat * (I1 - 2 * eps_x)
                                     * (6 * I2 + 2 * I2 * log(I2) - 7 * I2 * I2
                                        - 6 * I2 * I2 * log(I2) + 1))
                                  / (I2 * eps_x * eps_x);
#elif(RANK == 2)
                double lambda20 =
                    Kappa
                    * (4 * I1 * dHat * dHat * (I1 - 2 * eps_x)
                       * (4 * I2 + log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                          + 6 * I2 * log(I2) - 2 * I2 * I2
                          + I2 * log(I2) * log(I2) - 7 * I2 * I2 * log(I2) - 2))
                    / (I2 * (eps_x * eps_x));
#elif(RANK == 4)
                double lambda20 =
                    Kappa
                    * (4 * I1 * dHat * dHat * log(I2) * log(I2) * (I1 - 2 * eps_x)
                       * (24 * I2 + 2 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                          + 12 * I2 * log(I2) - 12 * I2 * I2
                          + I2 * log(I2) * log(I2) - 14 * I2 * I2 * log(I2) - 12))
                    / (I2 * (eps_x * eps_x));
#elif(RANK == 6)
                double lambda20 =
                    Kappa
                    * (4 * I1 * dHat * dHat * pow(log(I2), 4) * (I1 - 2 * eps_x)
                       * (60 * I2 + 3 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                          + 18 * I2 * log(I2) - 30 * I2 * I2
                          + I2 * log(I2) * log(I2) - 21 * I2 * I2 * log(I2) - 30))
                    / (I2 * (eps_x * eps_x));
#endif

#if(RANK == 1)
                double lambdag1g = Kappa * 4 * c * F.m[2][2]
                                   * ((2 * dHat * dHat * (I1 - eps_x) * (I2 - 1)
                                       * (I2 + 2 * I2 * log(I2) - 1))
                                      / (I2 * eps_x * eps_x));
#elif(RANK == 2)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                                   * (4 * dHat * dHat * log(I2) * (I1 - eps_x)
                                      * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                                   / (I2 * (eps_x * eps_x));
#elif(RANK == 4)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                                   * (4 * dHat * dHat * pow(log(I2), 3) * (I1 - eps_x)
                                      * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                                   / (I2 * (eps_x * eps_x));
#elif(RANK == 6)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                                   * (4 * dHat * dHat * pow(log(I2), 5) * (I1 - eps_x)
                                      * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                                   / (I2 * (eps_x * eps_x));
#endif
                double  eigenValues[2];
                int     eigenNum = 0;
                double2 eigenVecs[2];
                __GEIGEN__::__makePD2x2(
                    lambda10, lambdag1g, lambdag1g, lambda20, eigenValues, eigenNum, eigenVecs);

                __GEIGEN__::Vector12 w2, w3;
                double             l2 = 0, l3 = 0;
#pragma unroll
                for(int z = 0; z < 12; z++)
                {
                    w2.v[z] = 0;
                    w3.v[z] = 0;
                }

                for(int i = 0; i < eigenNum; i++)
                {
                    if(eigenValues[i] > 0)
                    {
                        __GEIGEN__::Matrix3x3d eigenMatrix;
                        __GEIGEN__::__set_Mat_val(eigenMatrix,
                                                  0,
                                                  0,
                                                  0,
                                                  0,
                                                  eigenVecs[i].x,
                                                  0,
                                                  0,
                                                  0,
                                                  eigenVecs[i].y);

                        __GEIGEN__::Vector9 eigenMVec =
                            __GEIGEN__::__Mat3x3_to_vec9_double(eigenMatrix);

                        // rank component eigenValues[i] * (PFPx*eigenMVec) * (PFPx*eigenMVec)^T
                        if(i == 0)
                        {
                            l2 = eigenValues[0];
                            w2 = __GEIGEN__::__M12x9_v9_multiply(PFPx, eigenMVec);
                        }
                        else
                        {
                            l3 = eigenValues[i];
                            w3 = __GEIGEN__::__M12x9_v9_multiply(PFPx, eigenMVec);
                        }
                    }
                }

                int Hidx = matIndex[idx];  //int Hidx = gipc::ATOMIC_ADD(_cpNum + 4, 1);

                __storeHessian12RankN(H12x12, Hidx, lambda11, lambda12, l2, l3, w0, w1, w2, w3);
                D4Index[Hidx] =
                    make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
            }
            else
            {
#ifdef NEWF
                double dis;
                _d_PP(_vertexes[v0I], _vertexes[MMCVIDI.y], dis);
                dis                            = sqrt(dis);
                double              d_hat_sqrt = sqrt(dHat);
                __GEIGEN__::Vector6 PFPxT;
                pFpx_pp2(_vertexes[v0I], _vertexes[MMCVIDI.y], d_hat_sqrt, PFPxT);
                double I5  = pow(dis / d_hat_sqrt, 2);
                double fnn = dis / d_hat_sqrt;

#if(RANK == 1)
                double flatten_pk1 =
                    fnn * 2 * Kappa
                    * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5;
#elif(RANK == 2)
                double flatten_pk1 = fnn * 2
                                     * (2 * Kappa * dHat * dHat * log(I5)
                                        * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                                     / I5;
#elif(RANK == 3)
                double flatten_pk1 = fnn * -2
                                     * (Kappa * dHat * dHat * log(I5) * log(I5)
                                        * (I5 - 1) * (3 * I5 + 2 * I5 * log(I5) - 3))
                                     / I5;
#elif(RANK == 4)
                double flatten_pk1 =
                    fnn
                    * (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                    / I5;
#elif(RANK == 5)
                double flatten_pk1 =
                    fnn * -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                    / I5;
#elif(RANK == 6)
                double flatten_pk1 =
                    fnn
                    * (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                    / I5;
#endif

                __GEIGEN__::Vector6 gradient_vec =
                    __GEIGEN__::__s_vec6_multiply(PFPxT, flatten_pk1);

#else
                double3 v0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
                double3 Ds  = v0;
                double  dis = __GEIGEN__::__norm(v0);
                //if (dis > dHat_sqrt) return;
                double3 vec_normal =
                    __GEIGEN__::__normalized(make_double3(-v0.x, -v0.y, -v0.z));
                double3 target = make_double3(0, 1, 0);
                double3 vec    = __GEIGEN__::__v_vec_cross(vec_normal, target);
                double  cos    = __GEIGEN__::__v_vec_dot(vec_normal, target);
                __GEIGEN__::Matrix3x3d rotation;
                __GEIGEN__::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);
                __GEIGEN__::Vector6 PDmPx;
                if(cos + 1 == 0)
                {
                    rotation.m[0][0] = -1;
                    rotation.m[1][1] = -1;
                }
                else
                {
                    __GEIGEN__::Matrix3x3d cross_vec;
                    __GEIGEN__::__set_Mat_val(
                        cross_vec, 0, -vec.z, vec.y, vec.z, 0, -vec.x, -vec.y, vec.x, 0);

                    rotation = __GEIGEN__::__Mat_add(
                        rotation,
                        __GEIGEN__::__Mat_add(cross_vec,
                                              __GEIGEN__::__S_Mat_multiply(
                                                  __GEIGEN__::__M_Mat_multiply(cross_vec, cross_vec),
                                                  1.0 / (1 + cos))));
                }

                double3 pos0 = __GEIGEN__::__add(
                    _vertexes[v0I],
                    __GEIGEN__::__s_vec_multiply(vec_normal, dHat_sqrt - dis));
                double3 rotate_uv0 = __GEIGEN__::__M_v_multiply(rotation, pos0);
                double3 rotate_uv1 =
                    __GEIGEN__::__M_v_multiply(rotation, _vertexes[MMCVIDI.y]);

                double uv0 = rotate_uv0.y;
                double uv1 = rotate_uv1.y;

                double u0    = uv1 - uv0;
                double Dm    = u0;  //PFPx
                double DmInv = 1 / u0;

                double3 F  = __GEIGEN__::__s_vec_multiply(Ds, DmInv);
                double  I5 = __GEIGEN__::__squaredNorm(F);

                double3 tmp         = F;

#if(RANK == 1)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif(RANK == 2)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                        / I5);

#elif(RANK == 3)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                           * (3 * I5 + 2 * I5 * log(I5) - 3))
                        / I5);
#elif(RANK == 4)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                        / I5);
#elif(RANK == 5)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                           * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                        / I5);
#elif(RANK == 6)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                        / I5);
#endif
                __GEIGEN__::Matrix3x6d PFPx = __computePFDsPX3D_3x6_double(DmInv);

                __GEIGEN__::Vector6 gradient_vec =
                    __GEIGEN__::__M6x3_v3_multiply(__GEIGEN__::__Transpose3x6(PFPx), flatten_pk1);
#endif


                {
                    gipc::ATOMIC_ADD(&(_gradient[v0I].x), gradient_vec.v[0]);
                    gipc::ATOMIC_ADD(&(_gradient[v0I].y), gradient_vec.v[1]);
                    gipc::ATOMIC_ADD(&(_gradient[v0I].z), gradient_vec.v[2]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                }

#if(RANK == 1)
                double lambda0 = Kappa
                                 * (2 * dHat * dHat
                                    * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5
                                       - 6 * I5 * I5 * log(I5) + 1))
                                 / I5;
                if(dis * dis < gassThreshold * dHat)
                {
                    double lambda1 =
                        Kappa
                        * (2 * dHat * dHat
                           * (6 * gassThreshold + 2 * gassThreshold * log(gassThreshold)
                              - 7 * gassThreshold * gassThreshold
                              - 6 * gassThreshold * gassThreshold * log(gassThreshold) + 1))
                        / gassThreshold;
                    lambda0 = lambda1;
                }
#elif(RANK == 2)
                double lambda0 =
                    -(4 * Kappa * dHat * dHat
                      * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 6 * I5 * log(I5) - 2 * I5 * I5
                         + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                    / I5;
                if(dis * dis < gassThreshold * dHat)
                {
                    double lambda1 =
                        -(4 * Kappa * dHat * dHat
                          * (4 * gassThreshold + log(gassThreshold)
                             - 3 * gassThreshold * gassThreshold
                                   * log(gassThreshold) * log(gassThreshold)
                             + 6 * gassThreshold * log(gassThreshold) - 2 * gassThreshold * gassThreshold
                             + gassThreshold * log(gassThreshold) * log(gassThreshold)
                             - 7 * gassThreshold * gassThreshold * log(gassThreshold) - 2))
                        / gassThreshold;
                    lambda0 = lambda1;
                }
#elif(RANK == 3)
                double lambda0 =
                    (2 * Kappa * dHat * dHat * log(I5)
                     * (24 * I5 + 3 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                        + 18 * I5 * log(I5) - 12 * I5 * I5
                        + 2 * I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 12))
                    / I5;
#elif(RANK == 4)
                double lambda0 =
                    -(4 * Kappa * dHat * dHat * log(I5) * log(I5)
                      * (24 * I5 + 2 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 12 * I5 * log(I5) - 12 * I5 * I5
                         + I5 * log(I5) * log(I5) - 14 * I5 * I5 * log(I5) - 12))
                    / I5;
#elif(RANK == 5)
                double lambda0 =
                    (2 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * (80 * I5 + 5 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                        + 30 * I5 * log(I5) - 40 * I5 * I5
                        + 2 * I5 * log(I5) * log(I5) - 35 * I5 * I5 * log(I5) - 40))
                    / I5;
#elif(RANK == 6)
                double lambda0 =
                    -(4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                      * (60 * I5 + 3 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 18 * I5 * log(I5) - 30 * I5 * I5
                         + I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 30))
                    / I5;
#endif


#ifdef NEWF
                double                 H       = lambda0;
                __GEIGEN__::Matrix6x6d Hessian = __GEIGEN__::__s_M6x6_Multiply(
                    __GEIGEN__::__v6_vec6_toMat6x6(PFPxT, PFPxT), H);
#else
                double3 q0 = __GEIGEN__::__s_vec_multiply(F, 1 / sqrt(I5));

                __GEIGEN__::Matrix3x3d H =
                    __GEIGEN__::__S_Mat_multiply(__GEIGEN__::__v_vec_toMat(q0, q0),
                                                 lambda0);  //lambda0 * q0 * q0.transpose();

                __GEIGEN__::Matrix6x3d PFPxTransPos = __GEIGEN__::__Transpose3x6(PFPx);
                __GEIGEN__::Matrix6x6d Hessian = __GEIGEN__::__M6x3_M3x6_Multiply(
                    __GEIGEN__::__M6x3_M3x3_Multiply(PFPxTransPos, H), PFPx);
#endif
                int Hidx = matIndex[idx];  //int Hidx = gipc::ATOMIC_ADD(_cpNum + 2, 1);

                H6x6[Hidx]    = Hessian;
                D2Index[Hidx] = make_uint2(v0I, MMCVIDI.y);
            }
        }
        else if(MMCVIDI.w < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.x = v0I;
                MMCVIDI.w = -MMCVIDI.w - 1;
                double3 v0 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
                double3 v1 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]);
                double c  = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1)) /*/ __GEIGEN__::__norm(v0)*/;
                double I1 = c * c;
                if(I1 == 0)
                    return;
                double dis;
                _d_PE(_vertexes[MMCVIDI.x],
                      _vertexes[MMCVIDI.y],
                      _vertexes[MMCVIDI.z],
                      dis);
                double I2 = dis / dHat;
                dis       = sqrt(dis);

                __GEIGEN__::Matrix3x3d F;
                __GEIGEN__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
                double3 n1 = make_double3(0, 1, 0);
                double3 n2 = make_double3(0, 0, 1);

                double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                            _rest_vertexes[MMCVIDI.w],
                                            _rest_vertexes[MMCVIDI.y],
                                            _rest_vertexes[MMCVIDI.z]);

                __GEIGEN__::Matrix3x3d g1, g2;

                __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(n1, n1);
                __GEIGEN__::__M_Mat_multiply(F, nn, g1);
                nn = __GEIGEN__::__v_vec_toMat(n2, n2);
                __GEIGEN__::__M_Mat_multiply(F, nn, g2);

                __GEIGEN__::Vector9 flatten_g1 = __GEIGEN__::__Mat3x3_to_vec9_double(g1);
                __GEIGEN__::Vector9 flatten_g2 = __GEIGEN__::__Mat3x3_to_vec9_double(g2);

                __GEIGEN__::Matrix12x9d PFPx;
                pFpx_ppe(_vertexes[MMCVIDI.x],
                         _vertexes[MMCVIDI.y],
                         _vertexes[MMCVIDI.z],
                         _vertexes[MMCVIDI.w],
                         dHat_sqrt,
                         PFPx);

#if(RANK == 1)
                double p1 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
                double p2 = Kappa * 2
                            * (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1)
                               * (I2 + 2 * I2 * log(I2) - 1))
                            / (I2 * eps_x * eps_x);
#elif(RANK == 2)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * log(I2) * log(I2)
                               * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                            / (I2 * (eps_x * eps_x));
#elif(RANK == 4)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x)
                               * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                            / (I2 * (eps_x * eps_x));
#elif(RANK == 6)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x)
                               * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                            / (I2 * (eps_x * eps_x));
#endif
                __GEIGEN__::Vector9 flatten_pk1 =
                    __GEIGEN__::__add9(__GEIGEN__::__s_vec9_multiply(flatten_g1, p1),
                                       __GEIGEN__::__s_vec9_multiply(flatten_g2, p2));
                __GEIGEN__::Vector12 gradient_vec =
                    __GEIGEN__::__M12x9_v9_multiply(PFPx, flatten_pk1);

                {
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
                }

#if(RANK == 1)
                double lambda10 =
                    Kappa * (4 * dHat * dHat * log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                    / (eps_x * eps_x);
                double lambda11 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
                double lambda12 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
#elif(RANK == 2)
                double lambda10 = -Kappa
                                  * (4 * dHat * dHat * log(I2) * log(I2)
                                     * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                                  / (eps_x * eps_x);
                double lambda11 = -Kappa
                                  * (4 * dHat * dHat * log(I2) * log(I2)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
                double lambda12 = -Kappa
                                  * (4 * dHat * dHat * log(I2) * log(I2)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
#elif(RANK == 4)
                double lambda10 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 4)
                                     * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                                  / (eps_x * eps_x);
                double lambda11 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 4)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
                double lambda12 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 4)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
#elif(RANK == 6)
                double lambda10 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 6)
                                     * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x))
                                  / (eps_x * eps_x);
                double lambda11 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 6)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
                double lambda12 = -Kappa
                                  * (4 * dHat * dHat * pow(log(I2), 6)
                                     * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                                  / (eps_x * eps_x);
#endif
                __GEIGEN__::Matrix3x3d Tx, Ty, Tz;
                __GEIGEN__::__set_Mat_val(Tx, 0, 0, 0, 0, 0, 1, 0, -1, 0);
                __GEIGEN__::__set_Mat_val(Ty, 0, 0, -1, 0, 0, 0, 1, 0, 0);
                __GEIGEN__::__set_Mat_val(Tz, 0, 1, 0, -1, 0, 0, 0, 0, 0);

                __GEIGEN__::Vector9 q11 = __GEIGEN__::__Mat3x3_to_vec9_double(
                    __GEIGEN__::__M_Mat_multiply(Tx, g1));
                __GEIGEN__::__normalized_vec9_double(q11);
                __GEIGEN__::Vector9 q12 = __GEIGEN__::__Mat3x3_to_vec9_double(
                    __GEIGEN__::__M_Mat_multiply(Tz, g1));
                __GEIGEN__::__normalized_vec9_double(q12);

                // projectedH is never materialized: Hessian = PFPx*projectedH*PFPx^T
                //          = sum_k lambdas[k]*w_k*w_k^T with w_k = PFPx*q_k.
                __GEIGEN__::Vector12 w0 = __GEIGEN__::__M12x9_v9_multiply(PFPx, q11);
                __GEIGEN__::Vector12 w1 = __GEIGEN__::__M12x9_v9_multiply(PFPx, q12);

#if(RANK == 1)
                double lambda20 = -Kappa
                                  * (2 * I1 * dHat * dHat * (I1 - 2 * eps_x)
                                     * (6 * I2 + 2 * I2 * log(I2) - 7 * I2 * I2
                                        - 6 * I2 * I2 * log(I2) + 1))
                                  / (I2 * eps_x * eps_x);
#elif(RANK == 2)
                double lambda20 =
                    Kappa
                    * (4 * I1 * dHat * dHat * (I1 - 2 * eps_x)
                       * (4 * I2 + log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                          + 6 * I2 * log(I2) - 2 * I2 * I2
                          + I2 * log(I2) * log(I2) - 7 * I2 * I2 * log(I2) - 2))
                    / (I2 * (eps_x * eps_x));
#elif(RANK == 4)
                double lambda20 =
                    Kappa
                    * (4 * I1 * dHat * dHat * log(I2) * log(I2) * (I1 - 2 * eps_x)
                       * (24 * I2 + 2 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                          + 12 * I2 * log(I2) - 12 * I2 * I2
                          + I2 * log(I2) * log(I2) - 14 * I2 * I2 * log(I2) - 12))
                    / (I2 * (eps_x * eps_x));
#elif(RANK == 6)
                double lambda20 =
                    Kappa
                    * (4 * I1 * dHat * dHat * pow(log(I2), 4) * (I1 - 2 * eps_x)
                       * (60 * I2 + 3 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2)
                          + 18 * I2 * log(I2) - 30 * I2 * I2
                          + I2 * log(I2) * log(I2) - 21 * I2 * I2 * log(I2) - 30))
                    / (I2 * (eps_x * eps_x));
#endif

#if(RANK == 1)
                double lambdag1g = Kappa * 4 * c * F.m[2][2]
                                   * ((2 * dHat * dHat * (I1 - eps_x) * (I2 - 1)
                                       * (I2 + 2 * I2 * log(I2) - 1))
                                      / (I2 * eps_x * eps_x));
#elif(RANK == 2)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                                   * (4 * dHat * dHat * log(I2) * (I1 - eps_x)
                                      * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                                   / (I2 * (eps_x * eps_x));
#elif(RANK == 4)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                                   * (4 * dHat * dHat * pow(log(I2), 3) * (I1 - eps_x)
                                      * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                                   / (I2 * (eps_x * eps_x));
#elif(RANK == 6)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2]
                                   * (4 * dHat * dHat * pow(log(I2), 5) * (I1 - eps_x)
                                      * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                                   / (I2 * (eps_x * eps_x));
#endif
                double  eigenValues[2];
                int     eigenNum = 0;
                double2 eigenVecs[2];
                __GEIGEN__::__makePD2x2(
                    lambda10, lambdag1g, lambdag1g, lambda20, eigenValues, eigenNum, eigenVecs);

                __GEIGEN__::Vector12 w2, w3;
                double             l2 = 0, l3 = 0;
#pragma unroll
                for(int z = 0; z < 12; z++)
                {
                    w2.v[z] = 0;
                    w3.v[z] = 0;
                }

                for(int i = 0; i < eigenNum; i++)
                {
                    if(eigenValues[i] > 0)
                    {
                        __GEIGEN__::Matrix3x3d eigenMatrix;
                        __GEIGEN__::__set_Mat_val(eigenMatrix,
                                                  0,
                                                  0,
                                                  0,
                                                  0,
                                                  eigenVecs[i].x,
                                                  0,
                                                  0,
                                                  0,
                                                  eigenVecs[i].y);

                        __GEIGEN__::Vector9 eigenMVec =
                            __GEIGEN__::__Mat3x3_to_vec9_double(eigenMatrix);

                        // rank component eigenValues[i] * (PFPx*eigenMVec) * (PFPx*eigenMVec)^T
                        if(i == 0)
                        {
                            l2 = eigenValues[0];
                            w2 = __GEIGEN__::__M12x9_v9_multiply(PFPx, eigenMVec);
                        }
                        else
                        {
                            l3 = eigenValues[i];
                            w3 = __GEIGEN__::__M12x9_v9_multiply(PFPx, eigenMVec);
                        }
                    }
                }

                int Hidx = matIndex[idx];  //int Hidx = gipc::ATOMIC_ADD(_cpNum + 4, 1);

                __storeHessian12RankN(H12x12, Hidx, lambda11, lambda12, l2, l3, w0, w1, w2, w3);
                D4Index[Hidx] =
                    make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
            }
            else
            {
#ifdef NEWF
                double dis;
                _d_PE(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
                dis                               = sqrt(dis);
                double                 d_hat_sqrt = sqrt(dHat);
                __GEIGEN__::Matrix9x4d PFPxT;
                pFpx_pe2(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], d_hat_sqrt, PFPxT);
                double              I5 = pow(dis / d_hat_sqrt, 2);
                __GEIGEN__::Vector4 fnn;
                fnn.v[0] = fnn.v[1] = fnn.v[2] = 0;  // = fnn.v[3] = fnn.v[4] = 1;
                fnn.v[3]                       = dis / d_hat_sqrt;
                //__GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(fnn, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
                // Rank-1 factorization: q0 == e3, so H = lambda0 * e3 * e3^T and
                // Hessian = PFPxT * H * PFPxT^T = lambda0 * c3 * c3^T (c3 = column 3).
#if(RANK == 1)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif(RANK == 2)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                        / I5);
#elif(RANK == 3)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                           * (3 * I5 + 2 * I5 * log(I5) - 3))
                        / I5);
#elif(RANK == 4)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                        / I5);
#elif(RANK == 5)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                           * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                        / I5);
#elif(RANK == 6)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                        / I5);
#endif

                // flatten_pk1 has only v[3] nonzero: gradient = flatten_pk1.v[3] * c3.
                __GEIGEN__::Vector9 c3;
#pragma unroll
                for(int i = 0; i < 9; i++)
                    c3.v[i] = PFPxT.m[i][3];

                __GEIGEN__::Vector9 gradient_vec =
                    __GEIGEN__::__s_vec9_multiply(c3, flatten_pk1.v[3]);
#else

                double3 v0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
                double3 v1 = __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[v0I]);


                __GEIGEN__::Matrix3x2d Ds;
                __GEIGEN__::__set_Mat3x2_val_column(Ds, v0, v1);

                double3 triangle_normal =
                    __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(v0, v1));
                double3 target = make_double3(0, 1, 0);

                double3 vec = __GEIGEN__::__v_vec_cross(triangle_normal, target);
                double  cos = __GEIGEN__::__v_vec_dot(triangle_normal, target);

                double3 edge_normal = __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z]),
                    triangle_normal));
                double dis          = __GEIGEN__::__v_vec_dot(
                    __GEIGEN__::__minus(_vertexes[v0I], _vertexes[MMCVIDI.y]), edge_normal);

                __GEIGEN__::Matrix3x3d rotation;
                __GEIGEN__::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);

                __GEIGEN__::Matrix9x4d PDmPx;

                if(cos + 1 == 0)
                {
                    rotation.m[0][0] = -1;
                    rotation.m[1][1] = -1;
                }
                else
                {
                    __GEIGEN__::Matrix3x3d cross_vec;
                    __GEIGEN__::__set_Mat_val(
                        cross_vec, 0, -vec.z, vec.y, vec.z, 0, -vec.x, -vec.y, vec.x, 0);

                    rotation = __GEIGEN__::__Mat_add(
                        rotation,
                        __GEIGEN__::__Mat_add(cross_vec,
                                              __GEIGEN__::__S_Mat_multiply(
                                                  __GEIGEN__::__M_Mat_multiply(cross_vec, cross_vec),
                                                  1.0 / (1 + cos))));
                }

                double3 pos0 = __GEIGEN__::__add(
                    _vertexes[v0I],
                    __GEIGEN__::__s_vec_multiply(edge_normal, dHat_sqrt - dis));

                double3 rotate_uv0 = __GEIGEN__::__M_v_multiply(rotation, pos0);
                double3 rotate_uv1 =
                    __GEIGEN__::__M_v_multiply(rotation, _vertexes[MMCVIDI.y]);
                double3 rotate_uv2 =
                    __GEIGEN__::__M_v_multiply(rotation, _vertexes[MMCVIDI.z]);
                double3 rotate_normal = __GEIGEN__::__M_v_multiply(rotation, edge_normal);

                double2 uv0    = make_double2(rotate_uv0.x, rotate_uv0.z);
                double2 uv1    = make_double2(rotate_uv1.x, rotate_uv1.z);
                double2 uv2    = make_double2(rotate_uv2.x, rotate_uv2.z);
                double2 normal = make_double2(rotate_normal.x, rotate_normal.z);

                double2 u0 = __GEIGEN__::__minus_v2(uv1, uv0);
                double2 u1 = __GEIGEN__::__minus_v2(uv2, uv0);

                __GEIGEN__::Matrix2x2d Dm;

                __GEIGEN__::__set_Mat2x2_val_column(Dm, u0, u1);

                __GEIGEN__::Matrix2x2d DmInv;
                __GEIGEN__::__Inverse2x2(Dm, DmInv);

                __GEIGEN__::Matrix3x2d F = __GEIGEN__::__M3x2_M2x2_Multiply(Ds, DmInv);

                double3 FxN = __GEIGEN__::__M3x2_v2_multiply(F, normal);
                double  I5  = __GEIGEN__::__squaredNorm(FxN);

                __GEIGEN__::Matrix3x2d fnn;

                __GEIGEN__::Matrix2x2d nn = __GEIGEN__::__v2_vec2_toMat2x2(normal, normal);

                fnn = __GEIGEN__::__M3x2_M2x2_Multiply(F, nn);

                __GEIGEN__::Vector6 tmp = __GEIGEN__::__Mat3x2_to_vec6_double(fnn);

#if(RANK == 1)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif(RANK == 2)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                        / I5);
#elif(RANK == 3)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                           * (3 * I5 + 2 * I5 * log(I5) - 3))
                        / I5);
#elif(RANK == 4)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                        / I5);
#elif(RANK == 5)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                           * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                        / I5);
#elif(RANK == 6)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                        / I5);
#endif

                __GEIGEN__::Matrix6x9d PFPx = __computePFDsPX3D_6x9_double(DmInv);

                __GEIGEN__::Vector9 gradient_vec =
                    __GEIGEN__::__M9x6_v6_multiply(__GEIGEN__::__Transpose6x9(PFPx), flatten_pk1);
#endif

                {
                    gipc::ATOMIC_ADD(&(_gradient[v0I].x), gradient_vec.v[0]);
                    gipc::ATOMIC_ADD(&(_gradient[v0I].y), gradient_vec.v[1]);
                    gipc::ATOMIC_ADD(&(_gradient[v0I].z), gradient_vec.v[2]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
                }

#if(RANK == 1)
                double lambda0 = Kappa
                                 * (2 * dHat * dHat
                                    * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5
                                       - 6 * I5 * I5 * log(I5) + 1))
                                 / I5;
                if(dis * dis < gassThreshold * dHat)
                {
                    double lambda1 =
                        Kappa
                        * (2 * dHat * dHat
                           * (6 * gassThreshold + 2 * gassThreshold * log(gassThreshold)
                              - 7 * gassThreshold * gassThreshold
                              - 6 * gassThreshold * gassThreshold * log(gassThreshold) + 1))
                        / gassThreshold;
                    lambda0 = lambda1;
                }
#elif(RANK == 2)
                double lambda0 =
                    -(4 * Kappa * dHat * dHat
                      * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 6 * I5 * log(I5) - 2 * I5 * I5
                         + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                    / I5;
                if(dis * dis < gassThreshold * dHat)
                {
                    double lambda1 =
                        -(4 * Kappa * dHat * dHat
                          * (4 * gassThreshold + log(gassThreshold)
                             - 3 * gassThreshold * gassThreshold
                                   * log(gassThreshold) * log(gassThreshold)
                             + 6 * gassThreshold * log(gassThreshold) - 2 * gassThreshold * gassThreshold
                             + gassThreshold * log(gassThreshold) * log(gassThreshold)
                             - 7 * gassThreshold * gassThreshold * log(gassThreshold) - 2))
                        / gassThreshold;
                    lambda0 = lambda1;
                }
#elif(RANK == 3)
                double lambda0 =
                    (2 * Kappa * dHat * dHat * log(I5)
                     * (24 * I5 + 3 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                        + 18 * I5 * log(I5) - 12 * I5 * I5
                        + 2 * I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 12))
                    / I5;
#elif(RANK == 4)
                double lambda0 =
                    -(4 * Kappa * dHat * dHat * log(I5) * log(I5)
                      * (24 * I5 + 2 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 12 * I5 * log(I5) - 12 * I5 * I5
                         + I5 * log(I5) * log(I5) - 14 * I5 * I5 * log(I5) - 12))
                    / I5;
#elif(RANK == 5)
                double lambda0 =
                    (2 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * (80 * I5 + 5 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                        + 30 * I5 * log(I5) - 40 * I5 * I5
                        + 2 * I5 * log(I5) * log(I5) - 35 * I5 * I5 * log(I5) - 40))
                    / I5;
#elif(RANK == 6)
                double lambda0 =
                    -(4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                      * (60 * I5 + 3 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 18 * I5 * log(I5) - 30 * I5 * I5
                         + I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 30))
                    / I5;
#endif


#ifndef NEWF

                __GEIGEN__::Vector6 q0 = __GEIGEN__::__Mat3x2_to_vec6_double(fnn);

                q0 = __GEIGEN__::__s_vec6_multiply(q0, 1.0 / sqrt(I5));

                __GEIGEN__::Matrix6x6d H;
                __GEIGEN__::__init_Mat6x6(H, 0);

                H = __GEIGEN__::__S_Mat6x6_multiply(
                    __GEIGEN__::__v6_vec6_toMat6x6(q0, q0), lambda0);

                __GEIGEN__::Matrix9x6d PFPxTransPos = __GEIGEN__::__Transpose6x9(PFPx);
                __GEIGEN__::Matrix9x9d Hessian = __GEIGEN__::__M9x6_M6x9_Multiply(
                    __GEIGEN__::__M9x6_M6x6_Multiply(PFPxTransPos, H), PFPx);
#endif
                int Hidx = matIndex[idx];  //int Hidx = gipc::ATOMIC_ADD(_cpNum + 3, 1);

#ifdef NEWF
                __storeHessian9Rank1(H9x9, Hidx, lambda0, c3);
#else
                H9x9[Hidx]    = Hessian;
#endif
                D3Index[Hidx] = make_uint3(v0I, MMCVIDI.y, MMCVIDI.z);
            }
        }
        else
        {
#ifdef NEWF
            double dis;
            _d_PT(_vertexes[v0I],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            dis                                = sqrt(dis);
            double                  d_hat_sqrt = sqrt(dHat);
            __GEIGEN__::Matrix12x9d PFPxT;
            pFpx_pt2(_vertexes[v0I],
                     _vertexes[MMCVIDI.y],
                     _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w],
                     d_hat_sqrt,
                     PFPxT);
            double              I5 = pow(dis / d_hat_sqrt, 2);
            __GEIGEN__::Vector9 tmp;
            tmp.v[0] = tmp.v[1] = tmp.v[2] = tmp.v[3] = tmp.v[4] = tmp.v[5] =
                tmp.v[6] = tmp.v[7] = 0;
            tmp.v[8]                = dis / d_hat_sqrt;

            // Rank-1 factorization: q0 == e8, see the EE branch above.
#else
            double3 v0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
            double3 v1 = __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[v0I]);
            double3 v2 = __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[v0I]);

            __GEIGEN__::Matrix3x3d Ds;
            __GEIGEN__::__set_Mat_val_column(Ds, v0, v1, v2);

            double3 normal = __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.y])));
            double  dis    = __GEIGEN__::__v_vec_dot(v0, normal);
            //if (abs(dis) > dHat_sqrt) return;
            __GEIGEN__::Matrix12x9d PDmPx;
            //bool is_flip = false;

            if(dis > 0)
            {
                //is_flip = true;
                normal = make_double3(-normal.x, -normal.y, -normal.z);
                //pDmpx_pt_flip(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dHat_sqrt, PDmPx);
                //printf("dHat_sqrt = %f,   dis = %f\n", dHat_sqrt, dis);
            }
            else
            {
                dis = -dis;
                //pDmpx_pt(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dHat_sqrt, PDmPx);
                //printf("dHat_sqrt = %f,   dis = %f\n", dHat_sqrt, dis);
            }

            double3 pos0 = __GEIGEN__::__add(
                _vertexes[v0I], __GEIGEN__::__s_vec_multiply(normal, dHat_sqrt - dis));


            double3 u0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], pos0);
            double3 u1 = __GEIGEN__::__minus(_vertexes[MMCVIDI.z], pos0);
            double3 u2 = __GEIGEN__::__minus(_vertexes[MMCVIDI.w], pos0);

            __GEIGEN__::Matrix3x3d Dm, DmInv;
            __GEIGEN__::__set_Mat_val_column(Dm, u0, u1, u2);

            __GEIGEN__::__Inverse(Dm, DmInv);

            __GEIGEN__::Matrix3x3d F;  //, Ftest;
            __GEIGEN__::__M_Mat_multiply(Ds, DmInv, F);
            //__GEIGEN__::__M_Mat_multiply(Dm, DmInv, Ftest);

            double3 FxN = __GEIGEN__::__M_v_multiply(F, normal);
            double  I5  = __GEIGEN__::__squaredNorm(FxN);

            //printf("I5 = %f,   dist/dHat_sqrt = %f\n", I5, (dis / dHat_sqrt)* (dis / dHat_sqrt));


            __GEIGEN__::Matrix9x12d PFPx = __computePFDsPX3D_double(DmInv);

            __GEIGEN__::Matrix3x3d fnn;

            __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(normal, normal);

            __GEIGEN__::__M_Mat_multiply(F, nn, fnn);

            __GEIGEN__::Vector9 tmp = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);
#endif
#if(RANK == 1)
            double lambda0 =
                Kappa
                * (2 * dHat * dHat
                   * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1))
                / I5;
            if(dis * dis < gassThreshold * dHat)
            {
                double lambda1 =
                    Kappa
                    * (2 * dHat * dHat
                       * (6 * gassThreshold + 2 * gassThreshold * log(gassThreshold)
                          - 7 * gassThreshold * gassThreshold
                          - 6 * gassThreshold * gassThreshold * log(gassThreshold) + 1))
                    / gassThreshold;
                lambda0 = lambda1;
            }
#elif(RANK == 2)
            double              lambda0 =
                -(4 * Kappa * dHat * dHat
                  * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 6 * I5 * log(I5)
                     - 2 * I5 * I5 + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                / I5;
            if(dis * dis < gassThreshold * dHat)
            {
                double lambda1 =
                    -(4 * Kappa * dHat * dHat
                      * (4 * gassThreshold + log(gassThreshold)
                         - 3 * gassThreshold * gassThreshold * log(gassThreshold) * log(gassThreshold)
                         + 6 * gassThreshold * log(gassThreshold) - 2 * gassThreshold * gassThreshold
                         + gassThreshold * log(gassThreshold) * log(gassThreshold)
                         - 7 * gassThreshold * gassThreshold * log(gassThreshold) - 2))
                    / gassThreshold;
                lambda0 = lambda1;
            }
#elif(RANK == 3)
            double lambda0 =
                (2 * Kappa * dHat * dHat * log(I5)
                 * (24 * I5 + 3 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                    + 18 * I5 * log(I5) - 12 * I5 * I5
                    + 2 * I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 12))
                / I5;
#elif(RANK == 4)
            double lambda0 =
                -(4 * Kappa * dHat * dHat * log(I5) * log(I5)
                  * (24 * I5 + 2 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                     + 12 * I5 * log(I5) - 12 * I5 * I5 + I5 * log(I5) * log(I5)
                     - 14 * I5 * I5 * log(I5) - 12))
                / I5;
#elif(RANK == 5)
            double lambda0 =
                (2 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                 * (80 * I5 + 5 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5)
                    + 30 * I5 * log(I5) - 40 * I5 * I5
                    + 2 * I5 * log(I5) * log(I5) - 35 * I5 * I5 * log(I5) - 40))
                / I5;
#elif(RANK == 6)
            double lambda0 =
                -(4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                  * (60 * I5 + 3 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                     + 18 * I5 * log(I5) - 30 * I5 * I5 + I5 * log(I5) * log(I5)
                     - 21 * I5 * I5 * log(I5) - 30))
                / I5;
#endif

#if(RANK == 1)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif(RANK == 2)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);
#elif(RANK == 3)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                       * (3 * I5 + 2 * I5 * log(I5) - 3))
                    / I5);
#elif(RANK == 4)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                 * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                    / I5);
#elif(RANK == 5)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                    / I5);
#elif(RANK == 6)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                 * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                    / I5);
#endif

#ifdef NEWF
            // flatten_pk1 has only v[8] nonzero: gradient = flatten_pk1.v[8] * c8,
            // Hessian = lambda0 * c8 * c8^T with c8 = column 8 of PFPxT.
            __GEIGEN__::Vector12 c8;
#pragma unroll
            for(int i = 0; i < 12; i++)
                c8.v[i] = PFPxT.m[i][8];

            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__s_vec12_multiply(c8, flatten_pk1.v[8]);
#else
            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply(__GEIGEN__::__Transpose9x12(PFPx), flatten_pk1);
#endif

            gipc::ATOMIC_ADD(&(_gradient[v0I].x), gradient_vec.v[0]);
            gipc::ATOMIC_ADD(&(_gradient[v0I].y), gradient_vec.v[1]);
            gipc::ATOMIC_ADD(&(_gradient[v0I].z), gradient_vec.v[2]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);

#ifndef NEWF

            //__GEIGEN__::Matrix3x3d Q0;

            //__GEIGEN__::Matrix3x3d fnn;

            //__GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(normal, normal);

            //__GEIGEN__::__M_Mat_multiply(F, nn, fnn);

            __GEIGEN__::Vector9 q0 = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);

            q0 = __GEIGEN__::__s_vec9_multiply(q0, 1.0 / sqrt(I5));

            __GEIGEN__::Matrix9x9d H = __GEIGEN__::__S_Mat9x9_multiply(
                __GEIGEN__::__v9_vec9_toMat9x9(q0, q0), lambda0);

            __GEIGEN__::Matrix12x9d PFPxTransPos = __GEIGEN__::__Transpose9x12(PFPx);
            __GEIGEN__::Matrix12x12d Hessian = __GEIGEN__::__M12x9_M9x12_Multiply(
                __GEIGEN__::__M12x9_M9x9_Multiply(PFPxTransPos, H), PFPx);
#endif

            int Hidx = matIndex[idx];  //int Hidx = gipc::ATOMIC_ADD(_cpNum + 4, 1);

#ifdef NEWF
            __storeHessian12Rank1(H12x12, Hidx, lambda0, c8);
#else
            H12x12[Hidx]  = Hessian;
#endif
            D4Index[Hidx] = make_uint4(v0I, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
        }
    }
}


__global__ void _calSelfCloseVal(const double3* _vertexes,
                                 const int4*    _collisionPair,
                                 int4*          _close_collisionPair,
                                 double*        _close_collisionVal,
                                 uint32_t*      _close_cpNum,
                                 double         dTol,
                                 int            number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4   MMCVIDI = _collisionPair[idx];
    double dist2   = _selfConstraintVal(_vertexes, MMCVIDI);
    if(dist2 < dTol)
    {
        int tidx                   = gipc::ATOMIC_ADD(_close_cpNum, 1);
        _close_collisionPair[tidx] = MMCVIDI;
        _close_collisionVal[tidx]  = dist2;
    }
}

__global__ void _checkSelfCloseVal(const double3* _vertexes,
                                   int*           _isChange,
                                   int4*          _close_collisionPair,
                                   double*        _close_collisionVal,
                                   int            number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4   MMCVIDI = _close_collisionPair[idx];
    double dist2   = _selfConstraintVal(_vertexes, MMCVIDI);
    if(dist2 < _close_collisionVal[idx])
    {
        *_isChange = 1;
    }
}


__global__ void _reduct_MSelfDist(const double3* _vertexes,
                                  int4*          _collisionPairs,
                                  double2*       _queue,
                                  int            number)
{
    int                       idof = blockIdx.x * blockDim.x;
    int                       idx  = threadIdx.x + idof;
    extern __shared__ double2 sdata[];

    if(idx >= number)
        return;
    int4    MMCVIDI = _collisionPairs[idx];
    double  tempv   = _selfConstraintVal(_vertexes, MMCVIDI);
    double2 temp    = make_double2(1.0 / tempv, tempv);
    __blockReduceMax2(temp, sdata, _queue, number);
}


__global__ void _calFrictionGradient_gd(const double3*        _vertexes,
                                        const double3*        _o_vertexes,
                                        const double3*        _normal,
                                        const uint32_t* _last_collisionPair_gd,
                                        double3*              _gradient,
                                        int                   number,
                                        double                dt,
                                        double                eps2,
                                        double*               lastH,
                                        double                coef)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double   eps    = sqrt(eps2);
    double3  normal = *_normal;
    uint32_t gidx   = _last_collisionPair_gd[idx];
    double3  Vdiff  = __GEIGEN__::__minus(_vertexes[gidx], _o_vertexes[gidx]);
    double3  VProj  = __GEIGEN__::__minus(
        Vdiff, __GEIGEN__::__s_vec_multiply(normal, __GEIGEN__::__v_vec_dot(Vdiff, normal)));
    double VProjMag2 = __GEIGEN__::__squaredNorm(VProj);
    if(VProjMag2 > eps2)
    {
        double3 gdf =
            __GEIGEN__::__s_vec_multiply(VProj, coef * lastH[idx] / sqrt(VProjMag2));
        /*gipc::ATOMIC_ADD(&(_gradient[gidx].x), gdf.x);
        gipc::ATOMIC_ADD(&(_gradient[gidx].y), gdf.y);
        gipc::ATOMIC_ADD(&(_gradient[gidx].z), gdf.z);*/
        _gradient[gidx] = __GEIGEN__::__add(_gradient[gidx], gdf);
    }
    else
    {
        double3 gdf = __GEIGEN__::__s_vec_multiply(VProj, coef * lastH[idx] / eps);
        /*gipc::ATOMIC_ADD(&(_gradient[gidx].x), gdf.x);
        gipc::ATOMIC_ADD(&(_gradient[gidx].y), gdf.y);
        gipc::ATOMIC_ADD(&(_gradient[gidx].z), gdf.z);*/
        _gradient[gidx] = __GEIGEN__::__add(_gradient[gidx], gdf);
    }
}

__global__ void _calFrictionGradient(const double3*    _vertexes,
                                     const double3*    _o_vertexes,
                                     const int4* _last_collisionPair,
                                     double3*          _gradient,
                                     int               number,
                                     double            dt,
                                     double2*          distCoord,
                                     __GEIGEN__::Matrix3x2d* tanBasis,
                                     double                  eps2,
                                     double*                 lastH,
                                     double                  coef)
{
    double eps = std::sqrt(eps2);
    int    idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4    MMCVIDI = _last_collisionPair[idx];
    double3 relDX3D;
    if(MMCVIDI.x >= 0)
    {
        Friction::computeRelDX_EE(
            __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
            __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
            __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]),
            distCoord[idx].x,
            distCoord[idx].y,
            relDX3D);

        __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
        double2 relDX       = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);
        double  relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
        if(relDXSqNorm > eps2)
        {
            relDX = __GEIGEN__::__s_vec_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
        }
        else
        {
            double f1_div_relDXNorm;
            Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
            relDX = __GEIGEN__::__s_vec_multiply(relDX, f1_div_relDXNorm);
        }
        __GEIGEN__::Vector12 TTTDX;
        Friction::liftRelDXTanToMesh_EE(
            relDX, tanBasis[idx], distCoord[idx].x, distCoord[idx].y, TTTDX);
        TTTDX = __GEIGEN__::__s_vec12_multiply(TTTDX, lastH[idx] * coef);
        {
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].x), TTTDX.v[0]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].y), TTTDX.v[1]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].z), TTTDX.v[2]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].x), TTTDX.v[3]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].y), TTTDX.v[4]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].z), TTTDX.v[5]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].x), TTTDX.v[6]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].y), TTTDX.v[7]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].z), TTTDX.v[8]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].x), TTTDX.v[9]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].y), TTTDX.v[10]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].z), TTTDX.v[11]);
        }
    }
    else
    {
        int v0I = -MMCVIDI.x - 1;
        if(MMCVIDI.z < 0)
        {
            MMCVIDI.x = v0I;

            Friction::computeRelDX_PP(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                relDX3D);

            __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
            double2 relDX       = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);
            double  relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
            if(relDXSqNorm > eps2)
            {
                relDX = __GEIGEN__::__s_vec_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
            }
            else
            {
                double f1_div_relDXNorm;
                Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                relDX = __GEIGEN__::__s_vec_multiply(relDX, f1_div_relDXNorm);
            }

            __GEIGEN__::Vector6 TTTDX;
            Friction::liftRelDXTanToMesh_PP(relDX, tanBasis[idx], TTTDX);
            TTTDX = __GEIGEN__::__s_vec6_multiply(TTTDX, lastH[idx] * coef);
            {
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].x), TTTDX.v[0]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].y), TTTDX.v[1]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].z), TTTDX.v[2]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].x), TTTDX.v[3]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].y), TTTDX.v[4]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].z), TTTDX.v[5]);
            }
        }
        else if(MMCVIDI.w < 0)
        {
            MMCVIDI.x = v0I;
            Friction::computeRelDX_PE(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                distCoord[idx].x,
                relDX3D);

            __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
            double2 relDX       = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);
            double  relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
            if(relDXSqNorm > eps2)
            {
                relDX = __GEIGEN__::__s_vec_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
            }
            else
            {
                double f1_div_relDXNorm;
                Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                relDX = __GEIGEN__::__s_vec_multiply(relDX, f1_div_relDXNorm);
            }
            __GEIGEN__::Vector9 TTTDX;
            Friction::liftRelDXTanToMesh_PE(relDX, tanBasis[idx], distCoord[idx].x, TTTDX);
            TTTDX = __GEIGEN__::__s_vec9_multiply(TTTDX, lastH[idx] * coef);
            {
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].x), TTTDX.v[0]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].y), TTTDX.v[1]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].z), TTTDX.v[2]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].x), TTTDX.v[3]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].y), TTTDX.v[4]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].z), TTTDX.v[5]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].x), TTTDX.v[6]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].y), TTTDX.v[7]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].z), TTTDX.v[8]);
            }
        }
        else
        {
            MMCVIDI.x = v0I;
            Friction::computeRelDX_PT(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]),
                distCoord[idx].x,
                distCoord[idx].y,
                relDX3D);

            __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
            double2 relDX = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);

            double relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
            if(relDXSqNorm > eps2)
            {
                relDX = __GEIGEN__::__s_vec_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
            }
            else
            {
                double f1_div_relDXNorm;
                Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                relDX = __GEIGEN__::__s_vec_multiply(relDX, f1_div_relDXNorm);
            }
            __GEIGEN__::Vector12 TTTDX;
            Friction::liftRelDXTanToMesh_PT(
                relDX, tanBasis[idx], distCoord[idx].x, distCoord[idx].y, TTTDX);
            TTTDX = __GEIGEN__::__s_vec12_multiply(TTTDX, lastH[idx] * coef);

            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].x), TTTDX.v[0]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].y), TTTDX.v[1]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].z), TTTDX.v[2]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].x), TTTDX.v[3]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].y), TTTDX.v[4]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].z), TTTDX.v[5]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].x), TTTDX.v[6]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].y), TTTDX.v[7]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].z), TTTDX.v[8]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].x), TTTDX.v[9]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].y), TTTDX.v[10]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].z), TTTDX.v[11]);
        }
    }
}

__global__ void _calBarrierGradient(const double3*    _vertexes,
                                    const double3*    _rest_vertexes,
                                    const int4* _collisionPair,
                                    double3*          _gradient,
                                    double            dHat,
                                    double            Kappa,
                                    int               number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4   MMCVIDI   = _collisionPair[idx];
    double dHat_sqrt = sqrt(dHat);
    //double dHat = dHat_sqrt * dHat_sqrt;
    //double Kappa = 1;
    if(MMCVIDI.x >= 0)
    {
        if(MMCVIDI.w >= 0)
        {
#ifdef NEWF
            double dis;
            _d_EE(_vertexes[MMCVIDI.x],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            dis                                = sqrt(dis);
            double                  d_hat_sqrt = sqrt(dHat);
            __GEIGEN__::Matrix12x9d PFPxT;
            pFpx_ee2(_vertexes[MMCVIDI.x],
                     _vertexes[MMCVIDI.y],
                     _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w],
                     d_hat_sqrt,
                     PFPxT);
            double              I5 = pow(dis / d_hat_sqrt, 2);
            __GEIGEN__::Vector9 tmp;
            tmp.v[0] = tmp.v[1] = tmp.v[2] = tmp.v[3] = tmp.v[4] = tmp.v[5] =
                tmp.v[6] = tmp.v[7] = 0;
            tmp.v[8]                = dis / d_hat_sqrt;
#else

            double3 v0 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
            double3 v1 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.x]);
            double3 v2 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
            __GEIGEN__::Matrix3x3d Ds;
            __GEIGEN__::__set_Mat_val_column(Ds, v0, v1, v2);
            double3 normal = __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(
                v0, __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z])));
            double  dis    = __GEIGEN__::__v_vec_dot(v1, normal);
            if(dis < 0)
            {
                normal = make_double3(-normal.x, -normal.y, -normal.z);
                dis    = -dis;
            }

            double3 pos2 =
                __GEIGEN__::__add(_vertexes[MMCVIDI.z],
                                  __GEIGEN__::__s_vec_multiply(normal, dHat_sqrt - dis));
            double3 pos3 =
                __GEIGEN__::__add(_vertexes[MMCVIDI.w],
                                  __GEIGEN__::__s_vec_multiply(normal, dHat_sqrt - dis));

            double3 u0 = v0;
            double3 u1 = __GEIGEN__::__minus(pos2, _vertexes[MMCVIDI.x]);
            double3 u2 = __GEIGEN__::__minus(pos3, _vertexes[MMCVIDI.x]);

            __GEIGEN__::Matrix3x3d Dm, DmInv;
            __GEIGEN__::__set_Mat_val_column(Dm, u0, u1, u2);

            __GEIGEN__::__Inverse(Dm, DmInv);

            __GEIGEN__::Matrix3x3d F;
            __GEIGEN__::__M_Mat_multiply(Ds, DmInv, F);

            double3 FxN = __GEIGEN__::__M_v_multiply(F, normal);
            double  I5  = __GEIGEN__::__squaredNorm(FxN);

            __GEIGEN__::Matrix9x12d PFPx = __computePFDsPX3D_double(DmInv);

            __GEIGEN__::Matrix3x3d fnn;

            __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(normal, normal);

            __GEIGEN__::__M_Mat_multiply(F, nn, fnn);

            __GEIGEN__::Vector9 tmp = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);

#endif

#if(RANK == 1)
            double judge =
                (2 * dHat * dHat
                 * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1))
                / I5;
            double judge2 = 2 * (dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1))
                            / I5 * dis / d_hat_sqrt;
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
            //if (dis*dis<1e-2*dHat)
            //flatten_pk1 = __GEIGEN__::__s_vec9_multiply(tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5 / (I5) /*/ (I5) / (I5)*/);
#elif(RANK == 2)
            //__GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(tmp, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);

            double judge = -(4 * dHat * dHat
                             * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                                + 6 * I5 * log(I5) - 2 * I5 * I5
                                + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                           / I5;
            double judge2 =
                2 * (2 * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                / I5 * dis / dHat_sqrt;
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);
            //if (dis*dis<1e-2*dHat)
            //flatten_pk1 = __GEIGEN__::__s_vec9_multiply(tmp, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5/I5);

#elif(RANK == 3)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                       * (3 * I5 + 2 * I5 * log(I5) - 3))
                    / I5);
#elif(RANK == 4)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                 * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                    / I5);
#elif(RANK == 5)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                    / I5);
#elif(RANK == 6)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                 * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                    / I5);
#endif

#ifdef NEWF
            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply((PFPxT), flatten_pk1);
#else

            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply(__GEIGEN__::__Transpose9x12(PFPx), flatten_pk1);
#endif

            {
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
            }
        }
        else
        {
            //return;
            MMCVIDI.w = -MMCVIDI.w - 1;
            double3 v0 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
            double3 v1 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z]);
            double c  = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1)) /*/ __GEIGEN__::__norm(v0)*/;
            double I1 = c * c;
            if(I1 == 0)
                return;
            double dis;
            _d_EE(_vertexes[MMCVIDI.x],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            double I2 = dis / dHat;
            dis       = sqrt(dis);

            __GEIGEN__::Matrix3x3d F;
            __GEIGEN__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
            double3 n1 = make_double3(0, 1, 0);
            double3 n2 = make_double3(0, 0, 1);

            double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                        _rest_vertexes[MMCVIDI.y],
                                        _rest_vertexes[MMCVIDI.z],
                                        _rest_vertexes[MMCVIDI.w]);

            __GEIGEN__::Matrix3x3d g1, g2;

            __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(n1, n1);
            __GEIGEN__::__M_Mat_multiply(F, nn, g1);
            nn = __GEIGEN__::__v_vec_toMat(n2, n2);
            __GEIGEN__::__M_Mat_multiply(F, nn, g2);

            __GEIGEN__::Vector9 flatten_g1 = __GEIGEN__::__Mat3x3_to_vec9_double(g1);
            __GEIGEN__::Vector9 flatten_g2 = __GEIGEN__::__Mat3x3_to_vec9_double(g2);

            __GEIGEN__::Matrix12x9d PFPx;
            pFpx_pee(_vertexes[MMCVIDI.x],
                     _vertexes[MMCVIDI.y],
                     _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w],
                     dHat_sqrt,
                     PFPx);


#if(RANK == 1)
            double p1 = Kappa * 2
                        * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                        / (eps_x * eps_x);
            double p2 = Kappa * 2
                        * (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1)
                           * (I2 + 2 * I2 * log(I2) - 1))
                        / (I2 * eps_x * eps_x);
#elif(RANK == 2)
            double p1 = -Kappa * 2
                        * (2 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x)
                           * (I2 - 1) * (I2 - 1))
                        / (eps_x * eps_x);
            double p2 = -Kappa * 2
                        * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x)
                           * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                        / (I2 * (eps_x * eps_x));
#elif(RANK == 4)
            double p1 = -Kappa * 2
                        * (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x)
                           * (I2 - 1) * (I2 - 1))
                        / (eps_x * eps_x);
            double p2 = -Kappa * 2
                        * (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x)
                           * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                        / (I2 * (eps_x * eps_x));
#elif(RANK == 6)
            double p1 = -Kappa * 2
                        * (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x)
                           * (I2 - 1) * (I2 - 1))
                        / (eps_x * eps_x);
            double p2 = -Kappa * 2
                        * (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x)
                           * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                        / (I2 * (eps_x * eps_x));
#endif


            __GEIGEN__::Vector9 flatten_pk1 =
                __GEIGEN__::__add9(__GEIGEN__::__s_vec9_multiply(flatten_g1, p1),
                                   __GEIGEN__::__s_vec9_multiply(flatten_g2, p2));
            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply(PFPx, flatten_pk1);

            {
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
                gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
            }
        }
    }
    else
    {
        int v0I = -MMCVIDI.x - 1;
        if(MMCVIDI.z < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.z = -MMCVIDI.z - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;
                MMCVIDI.x = v0I;
                double3 v0 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.x]);
                double3 v1 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.y]);
                double c  = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1)) /*/ __GEIGEN__::__norm(v0)*/;
                double I1 = c * c;
                if(I1 == 0)
                    return;
                double dis;
                _d_PP(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], dis);
                double I2 = dis / dHat;
                dis       = sqrt(dis);

                __GEIGEN__::Matrix3x3d F;
                __GEIGEN__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
                double3 n1 = make_double3(0, 1, 0);
                double3 n2 = make_double3(0, 0, 1);

                double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                            _rest_vertexes[MMCVIDI.z],
                                            _rest_vertexes[MMCVIDI.y],
                                            _rest_vertexes[MMCVIDI.w]);

                __GEIGEN__::Matrix3x3d g1, g2;

                __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(n1, n1);
                __GEIGEN__::__M_Mat_multiply(F, nn, g1);
                nn = __GEIGEN__::__v_vec_toMat(n2, n2);
                __GEIGEN__::__M_Mat_multiply(F, nn, g2);

                __GEIGEN__::Vector9 flatten_g1 = __GEIGEN__::__Mat3x3_to_vec9_double(g1);
                __GEIGEN__::Vector9 flatten_g2 = __GEIGEN__::__Mat3x3_to_vec9_double(g2);

                __GEIGEN__::Matrix12x9d PFPx;
                pFpx_ppp(_vertexes[MMCVIDI.x],
                         _vertexes[MMCVIDI.y],
                         _vertexes[MMCVIDI.z],
                         _vertexes[MMCVIDI.w],
                         dHat_sqrt,
                         PFPx);
#if(RANK == 1)
                double p1 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
                double p2 = Kappa * 2
                            * (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1)
                               * (I2 + 2 * I2 * log(I2) - 1))
                            / (I2 * eps_x * eps_x);
#elif(RANK == 2)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * log(I2) * log(I2)
                               * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                            / (I2 * (eps_x * eps_x));
#elif(RANK == 4)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x)
                               * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                            / (I2 * (eps_x * eps_x));
#elif(RANK == 6)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x)
                               * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                            / (I2 * (eps_x * eps_x));
#endif
                __GEIGEN__::Vector9 flatten_pk1 =
                    __GEIGEN__::__add9(__GEIGEN__::__s_vec9_multiply(flatten_g1, p1),
                                       __GEIGEN__::__s_vec9_multiply(flatten_g2, p2));
                __GEIGEN__::Vector12 gradient_vec =
                    __GEIGEN__::__M12x9_v9_multiply(PFPx, flatten_pk1);

                {
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
                }
            }
            else
            {
#ifdef NEWF
                double dis;
                _d_PP(_vertexes[v0I], _vertexes[MMCVIDI.y], dis);
                dis                            = sqrt(dis);
                double              d_hat_sqrt = sqrt(dHat);
                __GEIGEN__::Vector6 PFPxT;
                pFpx_pp2(_vertexes[v0I], _vertexes[MMCVIDI.y], d_hat_sqrt, PFPxT);
                double I5  = pow(dis / d_hat_sqrt, 2);
                double fnn = dis / d_hat_sqrt;

#if(RANK == 1)


                double judge = (2 * dHat * dHat
                                * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5
                                   - 6 * I5 * I5 * log(I5) + 1))
                               / I5;
                double judge2 =
                    2 * (dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1))
                    / I5 * dis / d_hat_sqrt;
                double flatten_pk1 =
                    fnn * 2 * Kappa
                    * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5;
                //if (dis*dis<1e-2*dHat)
                //flatten_pk1 = fnn * 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5 / (I5) /*/ (I5) / (I5)*/;
#elif(RANK == 2)
                //double flatten_pk1 = fnn * 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5;

                double judge =
                    -(4 * dHat * dHat
                      * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 6 * I5 * log(I5) - 2 * I5 * I5
                         + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                    / I5;
                double judge2 =
                    2 * (2 * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                    / I5 * dis / dHat_sqrt;
                double flatten_pk1 = fnn * 2
                                     * (2 * Kappa * dHat * dHat * log(I5)
                                        * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                                     / I5;
                //if (dis*dis<1e-2*dHat)
                //flatten_pk1 = fnn * 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5/I5;

#elif(RANK == 3)
                double flatten_pk1 = fnn * -2
                                     * (Kappa * dHat * dHat * log(I5) * log(I5)
                                        * (I5 - 1) * (3 * I5 + 2 * I5 * log(I5) - 3))
                                     / I5;
#elif(RANK == 4)
                double flatten_pk1 =
                    fnn
                    * (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                    / I5;
#elif(RANK == 5)
                double flatten_pk1 =
                    fnn * -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                    / I5;
#elif(RANK == 6)
                double flatten_pk1 =
                    fnn
                    * (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                    / I5;
#endif

                __GEIGEN__::Vector6 gradient_vec =
                    __GEIGEN__::__s_vec6_multiply(PFPxT, flatten_pk1);

#else
                double3 v0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
                double3 Ds  = v0;
                double  dis = __GEIGEN__::__norm(v0);
                //if (dis > dHat_sqrt) return;
                double3 vec_normal =
                    __GEIGEN__::__normalized(make_double3(-v0.x, -v0.y, -v0.z));
                double3 target = make_double3(0, 1, 0);
                double3 vec    = __GEIGEN__::__v_vec_cross(vec_normal, target);
                double  cos    = __GEIGEN__::__v_vec_dot(vec_normal, target);
                __GEIGEN__::Matrix3x3d rotation;
                __GEIGEN__::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);
                __GEIGEN__::Vector6 PDmPx;
                if(cos + 1 == 0)
                {
                    rotation.m[0][0] = -1;
                    rotation.m[1][1] = -1;
                }
                else
                {
                    __GEIGEN__::Matrix3x3d cross_vec;
                    __GEIGEN__::__set_Mat_val(
                        cross_vec, 0, -vec.z, vec.y, vec.z, 0, -vec.x, -vec.y, vec.x, 0);

                    rotation = __GEIGEN__::__Mat_add(
                        rotation,
                        __GEIGEN__::__Mat_add(cross_vec,
                                              __GEIGEN__::__S_Mat_multiply(
                                                  __GEIGEN__::__M_Mat_multiply(cross_vec, cross_vec),
                                                  1.0 / (1 + cos))));
                }

                double3 pos0 = __GEIGEN__::__add(
                    _vertexes[v0I],
                    __GEIGEN__::__s_vec_multiply(vec_normal, dHat_sqrt - dis));
                double3 rotate_uv0 = __GEIGEN__::__M_v_multiply(rotation, pos0);
                double3 rotate_uv1 =
                    __GEIGEN__::__M_v_multiply(rotation, _vertexes[MMCVIDI.y]);

                double uv0 = rotate_uv0.y;
                double uv1 = rotate_uv1.y;

                double u0    = uv1 - uv0;
                double Dm    = u0;  //PFPx
                double DmInv = 1 / u0;

                double3 F  = __GEIGEN__::__s_vec_multiply(Ds, DmInv);
                double  I5 = __GEIGEN__::__squaredNorm(F);

                double3 tmp         = F;

#if(RANK == 1)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif(RANK == 2)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                        / I5);
#elif(RANK == 3)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                           * (3 * I5 + 2 * I5 * log(I5) - 3))
                        / I5);
#elif(RANK == 4)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                        / I5);
#elif(RANK == 5)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                           * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                        / I5);
#elif(RANK == 6)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                        / I5);
#endif
                __GEIGEN__::Matrix3x6d PFPx = __computePFDsPX3D_3x6_double(DmInv);

                __GEIGEN__::Vector6 gradient_vec =
                    __GEIGEN__::__M6x3_v3_multiply(__GEIGEN__::__Transpose3x6(PFPx), flatten_pk1);
#endif


                {
                    gipc::ATOMIC_ADD(&(_gradient[v0I].x), gradient_vec.v[0]);
                    gipc::ATOMIC_ADD(&(_gradient[v0I].y), gradient_vec.v[1]);
                    gipc::ATOMIC_ADD(&(_gradient[v0I].z), gradient_vec.v[2]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                }
            }
        }
        else if(MMCVIDI.w < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.x = v0I;
                MMCVIDI.w = -MMCVIDI.w - 1;
                double3 v0 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
                double3 v1 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]);
                double c  = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1)) /*/ __GEIGEN__::__norm(v0)*/;
                double I1 = c * c;
                if(I1 == 0)
                    return;
                double dis;
                _d_PE(_vertexes[MMCVIDI.x],
                      _vertexes[MMCVIDI.y],
                      _vertexes[MMCVIDI.z],
                      dis);
                double I2 = dis / dHat;
                dis       = sqrt(dis);

                __GEIGEN__::Matrix3x3d F;
                __GEIGEN__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
                double3 n1 = make_double3(0, 1, 0);
                double3 n2 = make_double3(0, 0, 1);

                double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                            _rest_vertexes[MMCVIDI.w],
                                            _rest_vertexes[MMCVIDI.y],
                                            _rest_vertexes[MMCVIDI.z]);

                __GEIGEN__::Matrix3x3d g1, g2;

                __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(n1, n1);
                __GEIGEN__::__M_Mat_multiply(F, nn, g1);
                nn = __GEIGEN__::__v_vec_toMat(n2, n2);
                __GEIGEN__::__M_Mat_multiply(F, nn, g2);

                __GEIGEN__::Vector9 flatten_g1 = __GEIGEN__::__Mat3x3_to_vec9_double(g1);
                __GEIGEN__::Vector9 flatten_g2 = __GEIGEN__::__Mat3x3_to_vec9_double(g2);

                __GEIGEN__::Matrix12x9d PFPx;
                pFpx_ppe(_vertexes[MMCVIDI.x],
                         _vertexes[MMCVIDI.y],
                         _vertexes[MMCVIDI.z],
                         _vertexes[MMCVIDI.w],
                         dHat_sqrt,
                         PFPx);

#if(RANK == 1)
                double p1 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
                double p2 = Kappa * 2
                            * (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1)
                               * (I2 + 2 * I2 * log(I2) - 1))
                            / (I2 * eps_x * eps_x);
#elif(RANK == 2)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * log(I2) * log(I2)
                               * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                            / (I2 * (eps_x * eps_x));
#elif(RANK == 4)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x)
                               * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                            / (I2 * (eps_x * eps_x));
#elif(RANK == 6)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x)
                               * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                            / (I2 * (eps_x * eps_x));
#endif
                __GEIGEN__::Vector9 flatten_pk1 =
                    __GEIGEN__::__add9(__GEIGEN__::__s_vec9_multiply(flatten_g1, p1),
                                       __GEIGEN__::__s_vec9_multiply(flatten_g2, p2));
                __GEIGEN__::Vector12 gradient_vec =
                    __GEIGEN__::__M12x9_v9_multiply(PFPx, flatten_pk1);

                {
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
                }
            }
            else
            {
#ifdef NEWF
                double dis;
                _d_PE(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
                dis                               = sqrt(dis);
                double                 d_hat_sqrt = sqrt(dHat);
                __GEIGEN__::Matrix9x4d PFPxT;
                pFpx_pe2(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], d_hat_sqrt, PFPxT);
                double              I5 = pow(dis / d_hat_sqrt, 2);
                __GEIGEN__::Vector4 fnn;
                fnn.v[0] = fnn.v[1] = fnn.v[2] = 0;  // = fnn.v[3] = fnn.v[4] = 1;
                fnn.v[3]                       = dis / d_hat_sqrt;
                //__GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(fnn, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);

#if(RANK == 1)


                double judge = (2 * dHat * dHat
                                * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5
                                   - 6 * I5 * I5 * log(I5) + 1))
                               / I5;
                double judge2 =
                    2 * (dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1))
                    / I5 * dis / d_hat_sqrt;
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
                //if (dis*dis<1e-2*dHat)
                //flatten_pk1 = __GEIGEN__::__s_vec4_multiply(fnn, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5 / (I5) /*/ (I5) / (I5)*/);

#elif(RANK == 2)
                //__GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(fnn, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);

                double judge =
                    -(4 * dHat * dHat
                      * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 6 * I5 * log(I5) - 2 * I5 * I5
                         + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                    / I5;
                double judge2 =
                    2 * (2 * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                    / I5 * dis / dHat_sqrt;
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                        / I5);
                //if (dis*dis<1e-2*dHat)
                //flatten_pk1 = __GEIGEN__::__s_vec4_multiply(fnn, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5/I5);
#elif(RANK == 3)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                           * (3 * I5 + 2 * I5 * log(I5) - 3))
                        / I5);
#elif(RANK == 4)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                        / I5);
#elif(RANK == 5)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                           * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                        / I5);
#elif(RANK == 6)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                        / I5);
#endif

                __GEIGEN__::Vector9 gradient_vec =
                    __GEIGEN__::__M9x4_v4_multiply(PFPxT, flatten_pk1);
#else

                double3 v0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
                double3 v1 = __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[v0I]);


                __GEIGEN__::Matrix3x2d Ds;
                __GEIGEN__::__set_Mat3x2_val_column(Ds, v0, v1);

                double3 triangle_normal =
                    __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(v0, v1));
                double3 target = make_double3(0, 1, 0);

                double3 vec = __GEIGEN__::__v_vec_cross(triangle_normal, target);
                double  cos = __GEIGEN__::__v_vec_dot(triangle_normal, target);

                double3 edge_normal = __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z]),
                    triangle_normal));
                double dis          = __GEIGEN__::__v_vec_dot(
                    __GEIGEN__::__minus(_vertexes[v0I], _vertexes[MMCVIDI.y]), edge_normal);

                __GEIGEN__::Matrix3x3d rotation;
                __GEIGEN__::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);

                __GEIGEN__::Matrix9x4d PDmPx;

                if(cos + 1 == 0)
                {
                    rotation.m[0][0] = -1;
                    rotation.m[1][1] = -1;
                }
                else
                {
                    __GEIGEN__::Matrix3x3d cross_vec;
                    __GEIGEN__::__set_Mat_val(
                        cross_vec, 0, -vec.z, vec.y, vec.z, 0, -vec.x, -vec.y, vec.x, 0);

                    rotation = __GEIGEN__::__Mat_add(
                        rotation,
                        __GEIGEN__::__Mat_add(cross_vec,
                                              __GEIGEN__::__S_Mat_multiply(
                                                  __GEIGEN__::__M_Mat_multiply(cross_vec, cross_vec),
                                                  1.0 / (1 + cos))));
                }

                double3 pos0 = __GEIGEN__::__add(
                    _vertexes[v0I],
                    __GEIGEN__::__s_vec_multiply(edge_normal, dHat_sqrt - dis));

                double3 rotate_uv0 = __GEIGEN__::__M_v_multiply(rotation, pos0);
                double3 rotate_uv1 =
                    __GEIGEN__::__M_v_multiply(rotation, _vertexes[MMCVIDI.y]);
                double3 rotate_uv2 =
                    __GEIGEN__::__M_v_multiply(rotation, _vertexes[MMCVIDI.z]);
                double3 rotate_normal = __GEIGEN__::__M_v_multiply(rotation, edge_normal);

                double2 uv0    = make_double2(rotate_uv0.x, rotate_uv0.z);
                double2 uv1    = make_double2(rotate_uv1.x, rotate_uv1.z);
                double2 uv2    = make_double2(rotate_uv2.x, rotate_uv2.z);
                double2 normal = make_double2(rotate_normal.x, rotate_normal.z);

                double2 u0 = __GEIGEN__::__minus_v2(uv1, uv0);
                double2 u1 = __GEIGEN__::__minus_v2(uv2, uv0);

                __GEIGEN__::Matrix2x2d Dm;

                __GEIGEN__::__set_Mat2x2_val_column(Dm, u0, u1);

                __GEIGEN__::Matrix2x2d DmInv;
                __GEIGEN__::__Inverse2x2(Dm, DmInv);

                __GEIGEN__::Matrix3x2d F = __GEIGEN__::__M3x2_M2x2_Multiply(Ds, DmInv);

                double3 FxN = __GEIGEN__::__M3x2_v2_multiply(F, normal);
                double  I5  = __GEIGEN__::__squaredNorm(FxN);

                __GEIGEN__::Matrix3x2d fnn;

                __GEIGEN__::Matrix2x2d nn = __GEIGEN__::__v2_vec2_toMat2x2(normal, normal);

                fnn = __GEIGEN__::__M3x2_M2x2_Multiply(F, nn);

                __GEIGEN__::Vector6 tmp = __GEIGEN__::__Mat3x2_to_vec6_double(fnn);


#if(RANK == 1)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif(RANK == 2)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                        / I5);
#elif(RANK == 3)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                           * (3 * I5 + 2 * I5 * log(I5) - 3))
                        / I5);
#elif(RANK == 4)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                        / I5);
#elif(RANK == 5)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                           * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                        / I5);
#elif(RANK == 6)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                        / I5);
#endif

                __GEIGEN__::Matrix6x9d PFPx = __computePFDsPX3D_6x9_double(DmInv);

                __GEIGEN__::Vector9 gradient_vec =
                    __GEIGEN__::__M9x6_v6_multiply(__GEIGEN__::__Transpose6x9(PFPx), flatten_pk1);
#endif

                {
                    gipc::ATOMIC_ADD(&(_gradient[v0I].x), gradient_vec.v[0]);
                    gipc::ATOMIC_ADD(&(_gradient[v0I].y), gradient_vec.v[1]);
                    gipc::ATOMIC_ADD(&(_gradient[v0I].z), gradient_vec.v[2]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
                    gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
                }
            }
        }
        else
        {
#ifdef NEWF
            double dis;
            _d_PT(_vertexes[v0I],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            dis                                = sqrt(dis);
            double                  d_hat_sqrt = sqrt(dHat);
            __GEIGEN__::Matrix12x9d PFPxT;
            pFpx_pt2(_vertexes[v0I],
                     _vertexes[MMCVIDI.y],
                     _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w],
                     d_hat_sqrt,
                     PFPxT);
            double              I5 = pow(dis / d_hat_sqrt, 2);
            __GEIGEN__::Vector9 tmp;
            tmp.v[0] = tmp.v[1] = tmp.v[2] = tmp.v[3] = tmp.v[4] = tmp.v[5] =
                tmp.v[6] = tmp.v[7] = 0;
            tmp.v[8]                = dis / d_hat_sqrt;
#else
            double3 v0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
            double3 v1 = __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[v0I]);
            double3 v2 = __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[v0I]);

            __GEIGEN__::Matrix3x3d Ds;
            __GEIGEN__::__set_Mat_val_column(Ds, v0, v1, v2);

            double3 normal = __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.y])));
            double  dis    = __GEIGEN__::__v_vec_dot(v0, normal);
            //if (abs(dis) > dHat_sqrt) return;
            __GEIGEN__::Matrix12x9d PDmPx;
            //bool is_flip = false;

            if(dis > 0)
            {
                //is_flip = true;
                normal = make_double3(-normal.x, -normal.y, -normal.z);
                //pDmpx_pt_flip(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dHat_sqrt, PDmPx);
                //printf("dHat_sqrt = %f,   dis = %f\n", dHat_sqrt, dis);
            }
            else
            {
                dis = -dis;
                //pDmpx_pt(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dHat_sqrt, PDmPx);
                //printf("dHat_sqrt = %f,   dis = %f\n", dHat_sqrt, dis);
            }

            double3 pos0 = __GEIGEN__::__add(
                _vertexes[v0I], __GEIGEN__::__s_vec_multiply(normal, dHat_sqrt - dis));


            double3 u0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], pos0);
            double3 u1 = __GEIGEN__::__minus(_vertexes[MMCVIDI.z], pos0);
            double3 u2 = __GEIGEN__::__minus(_vertexes[MMCVIDI.w], pos0);

            __GEIGEN__::Matrix3x3d Dm, DmInv;
            __GEIGEN__::__set_Mat_val_column(Dm, u0, u1, u2);

            __GEIGEN__::__Inverse(Dm, DmInv);

            __GEIGEN__::Matrix3x3d F;  //, Ftest;
            __GEIGEN__::__M_Mat_multiply(Ds, DmInv, F);
            //__GEIGEN__::__M_Mat_multiply(Dm, DmInv, Ftest);

            double3 FxN = __GEIGEN__::__M_v_multiply(F, normal);
            double  I5  = __GEIGEN__::__squaredNorm(FxN);

            //printf("I5 = %f,   dist/dHat_sqrt = %f\n", I5, (dis / dHat_sqrt)* (dis / dHat_sqrt));


            __GEIGEN__::Matrix9x12d PFPx = __computePFDsPX3D_double(DmInv);

            __GEIGEN__::Matrix3x3d fnn;

            __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(normal, normal);

            __GEIGEN__::__M_Mat_multiply(F, nn, fnn);

            __GEIGEN__::Vector9 tmp = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);
#endif


#if(RANK == 1)


            double judge =
                (2 * dHat * dHat
                 * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1))
                / I5;
            double judge2 = 2 * (dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1))
                            / I5 * dis / d_hat_sqrt;
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
            //if (dis*dis<1e-2*dHat)
            //flatten_pk1 = __GEIGEN__::__s_vec9_multiply(tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5 / (I5) /*/ (I5) / (I5)*/);

#elif(RANK == 2)
            //__GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(tmp, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);

            double judge = -(4 * dHat * dHat
                             * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                                + 6 * I5 * log(I5) - 2 * I5 * I5
                                + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                           / I5;
            double judge2 =
                2 * (2 * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                / I5 * dis / dHat_sqrt;
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);
            //if (dis*dis<1e-2*dHat)
            //flatten_pk1 = __GEIGEN__::__s_vec9_multiply(tmp, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5/I5);
#elif(RANK == 3)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                       * (3 * I5 + 2 * I5 * log(I5) - 3))
                    / I5);
#elif(RANK == 4)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                 * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                    / I5);
#elif(RANK == 5)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                    / I5);
#elif(RANK == 6)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                 * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                    / I5);
#endif

#ifdef NEWF
            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply(PFPxT, flatten_pk1);
#else
            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply(__GEIGEN__::__Transpose9x12(PFPx), flatten_pk1);
#endif

            gipc::ATOMIC_ADD(&(_gradient[v0I].x), gradient_vec.v[0]);
            gipc::ATOMIC_ADD(&(_gradient[v0I].y), gradient_vec.v[1]);
            gipc::ATOMIC_ADD(&(_gradient[v0I].z), gradient_vec.v[2]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
            gipc::ATOMIC_ADD(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
        }
    }
}

__global__ void _calKineticGradient(
    double3* vertexes, double3* xTilta, double3* gradient, double* masses, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    double3 deltaX = __GEIGEN__::__minus(vertexes[idx], xTilta[idx]);
    //masses[idx] = 1;
    gradient[idx] = make_double3(
        deltaX.x * masses[idx], deltaX.y * masses[idx], deltaX.z * masses[idx]);
    //printf("%f  %f  %f\n", gradient[idx].x, gradient[idx].y, gradient[idx].z);
}

__global__ void _computeSoftConstraintGradientAndHessian(const double3* vertexes,
                                                         const double3* targetVert,
                                                         const uint32_t* targetInd,
                                                         double3*  gradient,
                                                         uint32_t* _gpNum,
                                                         __GEIGEN__::Matrix3x3d* H3x3,
                                                         uint32_t* D1Index,
                                                         double    motionRate,
                                                         double    rate,
                                                         int       number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    uint32_t vInd = targetInd[idx];
    double   x = vertexes[vInd].x, y = vertexes[vInd].y, z = vertexes[vInd].z,
           a = targetVert[idx].x, b = targetVert[idx].y, c = targetVert[idx].z;
    //double dis = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(vertexes[vInd], targetVert[idx]));
    //printf("%f\n", dis);
    double d = motionRate;
    {
        gipc::ATOMIC_ADD(&(gradient[vInd].x), d * rate * rate * (x - a));
        gipc::ATOMIC_ADD(&(gradient[vInd].y), d * rate * rate * (y - b));
        gipc::ATOMIC_ADD(&(gradient[vInd].z), d * rate * rate * (z - c));
    }
    __GEIGEN__::Matrix3x3d Hpg;
    Hpg.m[0][0]   = rate * rate * d;
    Hpg.m[0][1]   = 0;
    Hpg.m[0][2]   = 0;
    Hpg.m[1][0]   = 0;
    Hpg.m[1][1]   = rate * rate * d;
    Hpg.m[1][2]   = 0;
    Hpg.m[2][0]   = 0;
    Hpg.m[2][1]   = 0;
    Hpg.m[2][2]   = rate * rate * d;
    int pidx      = gipc::ATOMIC_ADD(_gpNum, 1);
    H3x3[pidx]    = Hpg;
    D1Index[pidx] = vInd;
    //_environment_collisionPair[gipc::ATOMIC_ADD(_gpNum, 1)] = surfVertIds[idx];
}

__global__ void _computeSoftConstraintGradient(const double3*  vertexes,
                                               const double3*  targetVert,
                                               const uint32_t* targetInd,
                                               double3*        gradient,
                                               double          motionRate,
                                               double          rate,
                                               int             number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    uint32_t vInd = targetInd[idx];
    double   x = vertexes[vInd].x, y = vertexes[vInd].y, z = vertexes[vInd].z,
           a = targetVert[idx].x, b = targetVert[idx].y, c = targetVert[idx].z;
    //double dis = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(vertexes[vInd], targetVert[idx]));
    //printf("%f\n", dis);
    double d = motionRate;
    {
        gipc::ATOMIC_ADD(&(gradient[vInd].x), d * rate * rate * (x - a));
        gipc::ATOMIC_ADD(&(gradient[vInd].y), d * rate * rate * (y - b));
        gipc::ATOMIC_ADD(&(gradient[vInd].z), d * rate * rate * (z - c));
    }
}

__global__ void _GroundCollisionDetect(const double3*  vertexes,
                                       const uint32_t* surfVertIds,
                                       const double*   g_offset,
                                       const double3*  g_normal,
                                       uint32_t* _environment_collisionPair,
                                       uint32_t* _gpNum,
                                       double    dHat,
                                       int       number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double dist = __GEIGEN__::__v_vec_dot(*g_normal, vertexes[surfVertIds[idx]]) - *g_offset;
    if(dist * dist > dHat)
        return;

    _environment_collisionPair[gipc::ATOMIC_ADD(_gpNum, 1)] = surfVertIds[idx];
}

__global__ void _computeGroundGradientAndHessian(const double3* vertexes,
                                                 const double*  g_offset,
                                                 const double3* g_normal,
                                                 const uint32_t* _environment_collisionPair,
                                                 double3*  gradient,
                                                 uint32_t* _gpNum,
                                                 __GEIGEN__::Matrix3x3d* H3x3,
                                                 uint32_t* D1Index,
                                                 double    dHat,
                                                 double    Kappa,
                                                 int       number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double3 normal = *g_normal;
    int     gidx   = _environment_collisionPair[idx];
    double  dist  = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double  dist2 = dist * dist;

    double t   = dist2 - dHat;
    double g_b = t * log(dist2 / dHat) * -2.0 - (t * t) / dist2;

    double H_b = (log(dist2 / dHat) * -2.0 - t * 4.0 / dist2)
                 + 1.0 / (dist2 * dist2) * (t * t);

    //printf("H_b   dist   g_b    is  %lf  %lf  %lf\n", H_b, dist2, g_b);

    double3 grad = __GEIGEN__::__s_vec_multiply(normal, Kappa * g_b * 2 * dist);

    {
        gipc::ATOMIC_ADD(&(gradient[gidx].x), grad.x);
        gipc::ATOMIC_ADD(&(gradient[gidx].y), grad.y);
        gipc::ATOMIC_ADD(&(gradient[gidx].z), grad.z);
    }

    double param = 4.0 * H_b * dist2 + 2.0 * g_b;
    if(param > 0)
    {
        __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(normal, normal);
        __GEIGEN__::Matrix3x3d Hpg = __GEIGEN__::__S_Mat_multiply(nn, Kappa * param);

        int pidx      = gipc::ATOMIC_ADD(_gpNum, 1);
        H3x3[pidx]    = Hpg;
        D1Index[pidx] = gidx;
    }
    //_environment_collisionPair[gipc::ATOMIC_ADD(_gpNum, 1)] = surfVertIds[idx];
}

__global__ void _computeGroundGradient(const double3* vertexes,
                                       const double*  g_offset,
                                       const double3* g_normal,
                                       const uint32_t* _environment_collisionPair,
                                       double3*        gradient,
                                       uint32_t*       _gpNum,
                                       __GEIGEN__::Matrix3x3d* H3x3,
                                       double                  dHat,
                                       double                  Kappa,
                                       int                     number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double3 normal = *g_normal;
    int     gidx   = _environment_collisionPair[idx];
    double  dist  = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double  dist2 = dist * dist;

    double t   = dist2 - dHat;
    double g_b = t * std::log(dist2 / dHat) * -2.0 - (t * t) / dist2;

    //double H_b = (std::log(dist2 / dHat) * -2.0 - t * 4.0 / dist2) + 1.0 / (dist2 * dist2) * (t * t);
    double3 grad = __GEIGEN__::__s_vec_multiply(normal, Kappa * g_b * 2 * dist);

    {
        gipc::ATOMIC_ADD(&(gradient[gidx].x), grad.x);
        gipc::ATOMIC_ADD(&(gradient[gidx].y), grad.y);
        gipc::ATOMIC_ADD(&(gradient[gidx].z), grad.z);
    }
}

__global__ void _computeGroundCloseVal(const double3* vertexes,
                                       const double*  g_offset,
                                       const double3* g_normal,
                                       const uint32_t* _environment_collisionPair,
                                       double          dTol,
                                       uint32_t*       _closeConstraintID,
                                       double*         _closeConstraintVal,
                                       uint32_t*       _close_gpNum,
                                       int             number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double3 normal = *g_normal;
    int     gidx   = _environment_collisionPair[idx];
    double  dist  = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double  dist2 = dist * dist;

    if(dist2 < dTol)
    {
        int tidx                  = gipc::ATOMIC_ADD(_close_gpNum, 1);
        _closeConstraintID[tidx]  = gidx;
        _closeConstraintVal[tidx] = dist2;
    }
}

__global__ void _checkGroundCloseVal(const double3* vertexes,
                                     const double*  g_offset,
                                     const double3* g_normal,
                                     int*           _isChange,
                                     uint32_t*      _closeConstraintID,
                                     double*        _closeConstraintVal,
                                     int            number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double3 normal = *g_normal;
    int     gidx   = _closeConstraintID[idx];
    double  dist  = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double  dist2 = dist * dist;

    if(dist2 < _closeConstraintVal[idx])
    {
        *_isChange = 1;
    }
}

__global__ void _reduct_MGroundDist(const double3* vertexes,
                                    const double*  g_offset,
                                    const double3* g_normal,
                                    uint32_t*      _environment_collisionPair,
                                    double2*       _queue,
                                    int            number)
{
    int                       idof = blockIdx.x * blockDim.x;
    int                       idx  = threadIdx.x + idof;
    extern __shared__ double2 sdata[];

    if(idx >= number)
        return;
    double3 normal = *g_normal;
    int     gidx   = _environment_collisionPair[idx];
    double  dist  = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double  tempv = dist * dist;
    double2 temp  = make_double2(1.0 / tempv, tempv);

    __blockReduceMax2(temp, sdata, _queue, number);
}


__global__ void _checkGroundIntersection(const double3* vertexes,
                                         const double*  g_offset,
                                         const double3* g_normal,
                                         const uint32_t* _environment_collisionPair,
                                         int*            _isIntersect,
                                         int             number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double3 normal = *g_normal;
    int     gidx   = _environment_collisionPair[idx];
    double  dist = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    //printf("%f  %f\n", *g_offset, dist);
    if(dist < 0)
        *_isIntersect = -1;
}

__global__ void _getFrictionEnergy_Reduction_3D(double*        squeue,
                                                const double3* vertexes,
                                                const double3* o_vertexes,
                                                const int4*    _collisionPair,
                                                int            cpNum,
                                                double         dt,
                                                const double2* distCoord,
                                                const __GEIGEN__::Matrix3x2d* tanBasis,
                                                const double* lastH,
                                                double        fricDHat,
                                                double        eps

)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = cpNum;
    if(idx >= numbers)
        return;

    double temp = __cal_Friction_energy(
        vertexes, o_vertexes, _collisionPair[idx], dt, distCoord[idx], tanBasis[idx], lastH[idx], fricDHat, eps);

    __blockReduceSum(temp, tep, squeue, numbers);
}


__global__ void _getFrictionEnergy_gd_Reduction_3D(double*        squeue,
                                                   const double3* vertexes,
                                                   const double3* o_vertexes,
                                                   const double3* _normal,
                                                   const uint32_t* _collisionPair_gd,
                                                   int             gpNum,
                                                   double          dt,
                                                   const double* lastH,
                                                   double        eps

)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = gpNum;
    if(idx >= numbers)
        return;

    double temp = __cal_Friction_gd_energy(
        vertexes, o_vertexes, _normal, _collisionPair_gd[idx], dt, lastH[idx], eps);

    __blockReduceSum(temp, tep, squeue, numbers);
}


__global__ void _computeGroundEnergy_Reduction(double*        squeue,
                                               const double3* vertexes,
                                               const double*  g_offset,
                                               const double3* g_normal,
                                               const uint32_t* _environment_collisionPair,
                                               double dHat,
                                               double Kappa,
                                               int    number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;

    double3 normal = *g_normal;
    int     gidx   = _environment_collisionPair[idx];
    double  dist  = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double  dist2 = dist * dist;
    double  temp  = -(dist2 - dHat) * (dist2 - dHat) * log(dist2 / dHat);

    __blockReduceSum(temp, tep, squeue, number);
}


__global__ void _reduct_min_groundTimeStep_to_double(const double3*  vertexes,
                                                     const uint32_t* surfVertIds,
                                                     const double*   g_offset,
                                                     const double3*  g_normal,
                                                     const double3*  moveDir,
                                                     double* minStepSizes,
                                                     double  slackness,
                                                     int     number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;
    int     svI    = surfVertIds[idx];
    double  temp   = 1.0;
    double3 normal = *g_normal;
    double  coef   = __GEIGEN__::__v_vec_dot(normal, moveDir[svI]);
    if(coef > 0.0)
    {
        double dist = __GEIGEN__::__v_vec_dot(normal, vertexes[svI]) - *g_offset;  //normal
        temp = coef / (dist * slackness);
        //printf("%f\n", temp);
    }
    /*if (blockIdx.x == 4) {
        printf("%f\n", temp);
    }
    gipc::SYNC_THREADS();*/
    //printf("%f\n", temp);
    __blockReduceMax(temp, tep, minStepSizes, number);
}


__global__ void _reduct_min_InjectiveTimeStep_to_double(const double3* vertexes,
                                                        const uint4* tetrahedra,
                                                        const double3* moveDir,
                                                        double* minStepSizes,
                                                        double  slackness,
                                                        double  errorRate,
                                                        int     number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;
    double ratio = 1 - slackness;

    double temp = 1.0
                  / _computeInjectiveStepSize_3d(vertexes,
                                                 moveDir,
                                                 tetrahedra[idx].x,
                                                 tetrahedra[idx].y,
                                                 tetrahedra[idx].z,
                                                 tetrahedra[idx].w,
                                                 ratio,
                                                 errorRate);

    __blockReduceMax(temp, tep, minStepSizes, number);
}


__global__ void _reduct_min_selfTimeStep_to_double(const double3* vertexes,
                                                   const int4* _ccd_collitionPairs,
                                                   const double3* moveDir,
                                                   double*        minStepSizes,
                                                   double         slackness,
                                                   int            number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;
    double temp         = 1.0;
    double CCDDistRatio = 1.0 - slackness;

    int4 MMCVIDI = _ccd_collitionPairs[idx];

    if(MMCVIDI.x < 0)
    {
        MMCVIDI.x = -MMCVIDI.x - 1;

        double temp1 =
            point_triangle_ccd(vertexes[MMCVIDI.x],
                               vertexes[MMCVIDI.y],
                               vertexes[MMCVIDI.z],
                               vertexes[MMCVIDI.w],
                               __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.x], -1),
                               __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.y], -1),
                               __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.z], -1),
                               __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.w], -1),
                               CCDDistRatio,
                               0);

        //double temp2 = doCCDVF(vertexes[MMCVIDI.x],
        //    vertexes[MMCVIDI.y],
        //    vertexes[MMCVIDI.z],
        //    vertexes[MMCVIDI.w],
        //    __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.x], -1),
        //    __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.y], -1),
        //    __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.z], -1),
        //    __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.w], -1), 1e-9, 0.2);

        temp = 1.0 / temp1;
    }
    else
    {
        temp = 1.0
               / edge_edge_ccd(vertexes[MMCVIDI.x],
                               vertexes[MMCVIDI.y],
                               vertexes[MMCVIDI.z],
                               vertexes[MMCVIDI.w],
                               __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.x], -1),
                               __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.y], -1),
                               __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.z], -1),
                               __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.w], -1),
                               CCDDistRatio,
                               0);
    }

    __blockReduceMax(temp, tep, minStepSizes, number);
}


__global__ void _reduct_max_cfl_to_double(const double3* moveDir,
                                          double*        max_double_val,
                                          uint32_t*      mSVI,
                                          int            number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;

    double temp = __GEIGEN__::__norm(moveDir[mSVI[idx]]);


    __blockReduceMax(temp, tep, max_double_val, number);
}


__global__ void _reduct_double3Sqn_to_double(const double3* A, double* D, int number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;

    double temp = __GEIGEN__::__squaredNorm(A[idx]);


    __blockReduceSum(temp, tep, D, number);
}


__global__ void _reduct_double3Dot_to_double(const double3* A, const double3* B, double* D, int number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;

    double temp = __GEIGEN__::__v_vec_dot(A[idx], B[idx]);


    __blockReduceSum(temp, tep, D, number);
}



__global__ void _getKineticEnergy_Reduction_3D(
    double3* _vertexes, double3* _xTilta, double* _energy, double* _masses, int number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;

    double temp =
        __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(_vertexes[idx], _xTilta[idx]))
        * _masses[idx] * 0.5;

    __blockReduceSum(temp, tep, _energy, number);
}



__global__ void _getBendingEnergy_Reduction(double*        squeue,
                                            const double3* vertexes,
                                            const double3* rest_vertexex,
                                            const uint2*   edges,
                                            const uint2*   edge_adj_vertex,
                                            int            edgesNum,
                                            double         bendStiff)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = edgesNum;
    if(idx >= numbers)
        return;

    //double temp = __cal_BaraffWitkinStretch_energy(vertexes, triangles[idx], triDmInverses[idx], area[idx], stretchStiff, shearStiff);
    // double temp = __cal_hc_cloth_energy(vertexes, triangles[idx], triDmInverses[idx], area[idx], stretchStiff, shearStiff);
    uint2   adj     = edge_adj_vertex[idx];
    double3 rest_x0 = rest_vertexex[edges[idx].x];
    double3 rest_x1 = rest_vertexex[edges[idx].y];
    double  length  = __GEIGEN__::__norm(__GEIGEN__::__minus(rest_x0, rest_x1));
    double  temp =
        __cal_bending_energy(vertexes, rest_vertexex, edges[idx], adj, length, bendStiff);
    //double temp = 0;
    //printf("%f    %f\n\n\n", lenRate, volRate);
    __blockReduceSum(temp, tep, squeue, numbers);
}



__global__ void _getFEMEnergy_Reduction_3D(double*        squeue,
                                           const double3* vertexes,
                                           const uint4*   tetrahedras,
                                           const __GEIGEN__::Matrix3x3d* DmInverses,
                                           const double*                 volume,
                                           int    tetrahedraNum,
                                           double lenRate,
                                           double volRate)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = tetrahedraNum;
    if(idx >= numbers)
        return;

#ifdef USE_SNK
    double temp = __cal_StabbleNHK_energy_3D(
        vertexes, tetrahedras[idx], DmInverses[idx], volume[idx], lenRate, volRate);
#else
    double temp = __cal_ARAP_energy_3D(
        vertexes, tetrahedras[idx], DmInverses[idx], volume[idx], lenRate);
#endif

    //printf("%f    %f\n\n\n", lenRate, volRate);
    __blockReduceSum(temp, tep, squeue, numbers);
}

__global__ void _computeSoftConstraintEnergy_Reduction(double*        squeue,
                                                       const double3* vertexes,
                                                       const double3* targetVert,
                                                       const uint32_t* targetInd,
                                                       double motionRate,
                                                       double rate,
                                                       int    number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= number)
        return;
    uint32_t vInd = targetInd[idx];
    double   dis  = __GEIGEN__::__squaredNorm(__GEIGEN__::__s_vec_multiply(
        __GEIGEN__::__minus(vertexes[vInd], targetVert[idx]), rate));
    double   d    = motionRate;
    double   temp = d * dis * 0.5;

    __blockReduceSum(temp, tep, squeue, number);
}

__global__ void _get_triangleFEMEnergy_Reduction_3D(double*        squeue,
                                                    const double3* vertexes,
                                                    const uint3*   triangles,
                                                    const __GEIGEN__::Matrix2x2d* triDmInverses,
                                                    const double* area,
                                                    int           trianglesNum,
                                                    double        stretchStiff,
                                                    double        shearStiff)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = trianglesNum;
    if(idx >= numbers)
        return;

    double temp = __cal_BaraffWitkinStretch_energy(
        vertexes, triangles[idx], triDmInverses[idx], area[idx], stretchStiff, shearStiff);


    //printf("%f    %f\n\n\n", lenRate, volRate);
    __blockReduceSum(temp, tep, squeue, numbers);
}

__global__ void _getRestStableNHKEnergy_Reduction_3D(double*       squeue,
                                                     const double* volume,
                                                     int    tetrahedraNum,
                                                     double lenRate,
                                                     double volRate)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = tetrahedraNum;
    if(idx >= numbers)
        return;

    double temp = ((0.5 * volRate * (3 * lenRate / 4 / volRate) * (3 * lenRate / 4 / volRate)
                    - 0.5 * lenRate * log(4.0)))
                  * volume[idx];

    __blockReduceSum(temp, tep, squeue, numbers);
}


__global__ void _getBarrierEnergy_Reduction_3D(double*        squeue,
                                               const double3* vertexes,
                                               const double3* rest_vertexes,
                                               int4*          _collisionPair,
                                               double         _Kappa,
                                               double         _dHat,
                                               int            cpNum)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = cpNum;
    if(idx >= numbers)
        return;

    double temp =
        __cal_Barrier_energy(vertexes, rest_vertexes, _collisionPair[idx], _Kappa, _dHat);

    __blockReduceSum(temp, tep, squeue, numbers);
}


__global__ void _getDeltaEnergy_Reduction(double* squeue, const double3* b, const double3* dx, int vertexNum)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = vertexNum;
    if(idx >= numbers)
        return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);

    double temp = __GEIGEN__::__v_vec_dot(b[idx], dx[idx]);

    __blockReduceSum(temp, tep, squeue, numbers);
}


__global__ void __add_reduction(double* mem, int numbers)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= numbers)
        return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    double temp = mem[idx];

    gipc::THREAD_FENCE();

    __blockReduceSum(temp, tep, mem, numbers);
}


__global__ void _stepForward(double3* _vertexes,
                             double3* _vertexesTemp,
                             double3* _moveDir,
                             int*     bType,
                             double   alpha,
                             bool     moveBoundary,
                             int      numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    if(abs(bType[idx]) != 1 || moveBoundary)
    {
        _vertexes[idx] =
            __GEIGEN__::__minus(_vertexesTemp[idx],
                                __GEIGEN__::__s_vec_multiply(_moveDir[idx], alpha));
    }
}

__global__ void _updateVelocities(double3* _vertexes,
                                  double3* _o_vertexes,
                                  double3* _velocities,
                                  int*     btype,
                                  double   ipc_dt,
                                  int      numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    if(btype[idx] == 0)
    {
        _velocities[idx] = __GEIGEN__::__s_vec_multiply(
            __GEIGEN__::__minus(_vertexes[idx], _o_vertexes[idx]), 1 / ipc_dt);
        _o_vertexes[idx] = _vertexes[idx];
    }
    else
    {
        _velocities[idx] = make_double3(0, 0, 0);
        _o_vertexes[idx] = _vertexes[idx];
    }
}

__global__ void _updateBoundary(double3* _vertexes, int* _btype, double3* _moveDir, double ipc_dt, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    if((_btype[idx]) == -1 || (_btype[idx]) == 1)
    {
        _vertexes[idx] = __GEIGEN__::__add(_vertexes[idx], _moveDir[idx]);
    }
}

__global__ void _updateBoundary2(int* _btype, __GEIGEN__::Matrix3x3d* _constraints, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    if((_btype[idx]) == 1)
    {
        _btype[idx] = 0;
        __GEIGEN__::__set_Mat_val(_constraints[idx], 1, 0, 0, 0, 1, 0, 0, 0, 1);
    }
}


__global__ void _updateBoundaryMoveDir(double3* _vertexes,
                                       int*     _btype,
                                       double3* _moveDir,
                                       double   ipc_dt,
                                       double   PI,
                                       double   alpha,
                                       int      numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    double                 angleX  = PI / 2.5 * ipc_dt * alpha;
    __GEIGEN__::Matrix3x3d rotationL, rotationR;
    __GEIGEN__::__set_Mat_val(
        rotationL, 1, 0, 0, 0, cos(angleX), sin(angleX), 0, -sin(angleX), cos(angleX));
    __GEIGEN__::__set_Mat_val(
        rotationR, 1, 0, 0, 0, cos(angleX), -sin(angleX), 0, sin(angleX), cos(angleX));

    _moveDir[idx] = make_double3(0, 0, 0);
    double mvl    = -1 * ipc_dt * alpha;
    if((_btype[idx]) == 1)
    {
        _moveDir[idx] = make_double3(mvl, 0, 0);  //__GEIGEN__::__minus(__GEIGEN__::__M_v_multiply(rotationL, _vertexes[idx]), _vertexes[idx]);
    }
    //    if ((_btype[idx]) > 0) {
    //        _moveDir[idx] = __GEIGEN__::__minus(__GEIGEN__::__M_v_multiply(rotationL, _vertexes[idx]), _vertexes[idx]);
    //    }
    //    if ((_btype[idx]) < 0) {
    //        _moveDir[idx] = __GEIGEN__::__minus(__GEIGEN__::__M_v_multiply(rotationR, _vertexes[idx]), _vertexes[idx]);
    //    }
}

__global__ void _computeXTilta(int*     _btype,
                               double3* _velocities,
                               double3* _o_vertexes,
                               double3* _xTilta,
                               double   ipc_dt,
                               double   rate,
                               int      numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    double3 gravityDtSq = make_double3(0, 0, 0);  //__GEIGEN__::__s_vec_multiply(make_double3(0, -9.8, 0), ipc_dt * ipc_dt);//Vector3d(0, gravity, 0) * IPC_dt * IPC_dt;
    if(_btype[idx] == 0)
    {
        gravityDtSq =
            __GEIGEN__::__s_vec_multiply(make_double3(0, -9.8, 0), ipc_dt * ipc_dt);
    }
    _xTilta[idx] = __GEIGEN__::__add(
        _o_vertexes[idx],
        __GEIGEN__::__add(__GEIGEN__::__s_vec_multiply(_velocities[idx], ipc_dt),
                          gravityDtSq));  //(mesh.V_prev[vI] + (mesh.velocities[vI] * IPC_dt + gravityDtSq));
}

__global__ void _updateSurfaces(uint32_t* sortIndex, uint3* _faces, int _offset_num, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    if(_faces[idx].x < _offset_num)
    {
        _faces[idx].x = sortIndex[_faces[idx].x];
    }
    else
    {
        _faces[idx].x = _faces[idx].x;
    }
    if(_faces[idx].y < _offset_num)
    {
        _faces[idx].y = sortIndex[_faces[idx].y];
    }
    else
    {
        _faces[idx].y = _faces[idx].y;
    }
    if(_faces[idx].z < _offset_num)
    {
        _faces[idx].z = sortIndex[_faces[idx].z];
    }
    else
    {
        _faces[idx].z = _faces[idx].z;
    }
    //printf("sorted face: %d  %d  %d\n", _faces[idx].x, _faces[idx].y, _faces[idx].z);
}

__global__ void _updateNeighborNum(unsigned int*   _neighborNumInit,
                                   unsigned int*   _neighborNum,
                                   const uint32_t* sortMapVertIndex,
                                   int             numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    _neighborNum[idx] = _neighborNumInit[sortMapVertIndex[idx]];
}

__global__ void _updateNeighborList(unsigned int*   _neighborListInit,
                                    unsigned int*   _neighborList,
                                    unsigned int*   _neighborNum,
                                    unsigned int*   _neighborStart,
                                    unsigned int*   _neighborStartTemp,
                                    const uint32_t* sortIndex,
                                    const uint32_t* sortMapVertIndex,
                                    int             numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    int startId   = _neighborStartTemp[idx];
    int o_startId = _neighborStart[sortIndex[idx]];
    int neiNum    = _neighborNum[idx];
    for(int i = 0; i < neiNum; i++)
    {
        _neighborList[startId + i] = sortMapVertIndex[_neighborListInit[o_startId + i]];
    }
    //_neighborStart[sortMapVertIndex[idx]] = startId;
    //_neighborNum[idx] = _neighborNum[sortMapVertIndex[idx]];
}

__global__ void _updateEdges(uint32_t* sortIndex, uint2* _edges, int _offset_num, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    if(_edges[idx].x < _offset_num)
    {
        _edges[idx].x = sortIndex[_edges[idx].x];
    }
    else
    {
        _edges[idx].x = _edges[idx].x;
    }
    if(_edges[idx].y < _offset_num)
    {
        _edges[idx].y = sortIndex[_edges[idx].y];
    }
    else
    {
        _edges[idx].y = _edges[idx].y;
    }
}

__global__ void _updateTriEdges_adjVerts(
    uint32_t* sortIndex, uint2* _edges, uint2* _adj_verts, int _offset_num, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    if(_edges[idx].x < _offset_num)
    {
        _edges[idx].x = sortIndex[_edges[idx].x];
    }
    else
    {
        _edges[idx].x = _edges[idx].x;
    }
    if(_edges[idx].y < _offset_num)
    {
        _edges[idx].y = sortIndex[_edges[idx].y];
    }
    else
    {
        _edges[idx].y = _edges[idx].y;
    }


    if(_adj_verts[idx].x < _offset_num)
    {
        _adj_verts[idx].x = sortIndex[_adj_verts[idx].x];
    }
    else
    {
        _adj_verts[idx].x = _adj_verts[idx].x;
    }
    if(_adj_verts[idx].y < _offset_num)
    {
        _adj_verts[idx].y = sortIndex[_adj_verts[idx].y];
    }
    else
    {
        _adj_verts[idx].y = _adj_verts[idx].y;
    }
}

__global__ void _updateSurfVerts(uint32_t* sortIndex, uint32_t* _sVerts, int _offset_num, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    if(_sVerts[idx] < _offset_num)
    {
        _sVerts[idx] = sortIndex[_sVerts[idx]];
    }
    else
    {
        _sVerts[idx] = _sVerts[idx];
    }
}

__global__ void _edgeTriIntersectionQuery(const int*     _btype,
                                          const double3* _vertexes,
                                          const uint2*   _edges,
                                          const uint3*   _faces,
                                          const AABB*    _edge_bvs,
                                          const Node*    _edge_nodes,
                                          int*           _isIntesect,
                                          double         dHat,
                                          int            edge_count,
                                          int            number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    if(edge_count <= 0)
        return;

    uint32_t  stack[65];
    uint32_t* stack_ptr = stack;
    *stack_ptr++        = 0;

    uint3 face = _faces[idx];
    //idx = idx + number - 1;


    AABB _bv;

    double3 _v = _vertexes[face.x];
    _bv.combines(_v.x, _v.y, _v.z);
    _v = _vertexes[face.y];
    _bv.combines(_v.x, _v.y, _v.z);
    _v = _vertexes[face.z];
    _bv.combines(_v.x, _v.y, _v.z);

    //uint32_t self_eid = _edge_nodes[idx].element_idx;
    //double bboxDiagSize2 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(_edge_bvs[0].upper, _edge_bvs[0].lower));
    //printf("%f\n", bboxDiagSize2);
    double gapl = 0;  //sqrt(dHat);
    const auto root_object = _edge_nodes[0].element_idx;
    if(root_object != 0xFFFFFFFF)
    {
        if(_overlap(_bv, _edge_bvs[0], gapl)
           && !(face.x == _edges[root_object].x || face.x == _edges[root_object].y
                || face.y == _edges[root_object].x || face.y == _edges[root_object].y
                || face.z == _edges[root_object].x || face.z == _edges[root_object].y)
           && !(_btype[face.x] >= 2 && _btype[face.y] >= 2 && _btype[face.z] >= 2
                && _btype[_edges[root_object].x] >= 2
                && _btype[_edges[root_object].y] >= 2)
           && segTriIntersect(_vertexes[_edges[root_object].x],
                              _vertexes[_edges[root_object].y],
                              _vertexes[face.x],
                              _vertexes[face.y],
                              _vertexes[face.z]))
            *_isIntesect = -1;
        return;
    }
    //double dHat = gapl * gapl;// *bboxDiagSize2;
    do
    {
        const uint32_t node_id = *--stack_ptr;
        const uint32_t L_idx   = _edge_nodes[node_id].left_idx;
        const uint32_t R_idx   = _edge_nodes[node_id].right_idx;

        if(_overlap(_bv, _edge_bvs[L_idx], gapl))
        {
            const auto obj_idx = _edge_nodes[L_idx].element_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(!(face.x == _edges[obj_idx].x || face.x == _edges[obj_idx].y
                     || face.y == _edges[obj_idx].x || face.y == _edges[obj_idx].y
                     || face.z == _edges[obj_idx].x || face.z == _edges[obj_idx].y))
                {
                    if(!(_btype[face.x] >= 2 && _btype[face.y] >= 2
                         && _btype[face.z] >= 2 && _btype[_edges[obj_idx].x] >= 2
                         && _btype[_edges[obj_idx].y] >= 2))
                        if(segTriIntersect(_vertexes[_edges[obj_idx].x],
                                           _vertexes[_edges[obj_idx].y],
                                           _vertexes[face.x],
                                           _vertexes[face.y],
                                           _vertexes[face.z]))
                        {
                            //gipc::ATOMIC_ADD(_isIntesect, -1);
                            *_isIntesect = -1;
                            printf("tri: %d %d %d,  edge: %d  %d\n",
                                   face.x,
                                   face.y,
                                   face.z,
                                   _edges[obj_idx].x,
                                   _edges[obj_idx].y);
                            return;
                        }
                }
            }
            else  // the node is not a leaf.
            {
                *stack_ptr++ = L_idx;
            }
        }
        if(_overlap(_bv, _edge_bvs[R_idx], gapl))
        {
            const auto obj_idx = _edge_nodes[R_idx].element_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(!(face.x == _edges[obj_idx].x || face.x == _edges[obj_idx].y
                     || face.y == _edges[obj_idx].x || face.y == _edges[obj_idx].y
                     || face.z == _edges[obj_idx].x || face.z == _edges[obj_idx].y))
                {
                    if(!(_btype[face.x] >= 2 && _btype[face.y] >= 2
                         && _btype[face.z] >= 2 && _btype[_edges[obj_idx].x] >= 2
                         && _btype[_edges[obj_idx].y] >= 2))
                        if(segTriIntersect(_vertexes[_edges[obj_idx].x],
                                           _vertexes[_edges[obj_idx].y],
                                           _vertexes[face.x],
                                           _vertexes[face.y],
                                           _vertexes[face.z]))
                        {
                            //gipc::ATOMIC_ADD(_isIntesect, -1);
                            *_isIntesect = -1;
                            printf("tri: %d %d %d,  edge: %d  %d\n",
                                   face.x,
                                   face.y,
                                   face.z,
                                   _edges[obj_idx].x,
                                   _edges[obj_idx].y);
                            return;
                        }
                }
            }
            else  // the node is not a leaf.
            {
                *stack_ptr++ = R_idx;
            }
        }
    } while(stack < stack_ptr);
}

__global__ void _calFrictionLastH_gd(const double3*        _vertexes,
                                     const double*         g_offset,
                                     const double3*        g_normal,
                                     const uint32_t* _collisionPair_environment,
                                     double*               lambda_lastH_gd,
                                     uint32_t* _collisionPair_last_gd,
                                     double    dHat,
                                     double    Kappa,
                                     int       number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    double3 normal = *g_normal;
    int     gidx   = _collisionPair_environment[idx];
    double  dist = __GEIGEN__::__v_vec_dot(normal, _vertexes[gidx]) - *g_offset;
    double  dist2 = dist * dist;

    double t   = dist2 - dHat;
    double g_b = t * log(dist2 / dHat) * -2.0 - (t * t) / dist2;

    lambda_lastH_gd[idx]        = -Kappa * 2.0 * sqrt(dist2) * g_b;
    _collisionPair_last_gd[idx] = gidx;
}

__global__ void _calFrictionLastH_DistAndTan(const double3*    _vertexes,
                                             const int4* _collisionPair,
                                             double*           lambda_lastH,
                                             double2*          distCoord,
                                             __GEIGEN__::Matrix3x2d* tanBasis,
                                             int4*     _collisionPair_last,
                                             double    dHat,
                                             double    Kappa,
                                             uint32_t* _cpNum_last,
                                             int       number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4   MMCVIDI = _collisionPair[idx];
    double dis;
    int    last_index = -1;
    if(MMCVIDI.x >= 0)
    {
        if(MMCVIDI.w >= 0)
        {
            last_index = gipc::ATOMIC_ADD(_cpNum_last, 1);
            gipc::ATOMIC_ADD(_cpNum_last + 4, 1);
            _d_EE(_vertexes[MMCVIDI.x],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            Friction::computeClosestPoint_EE(_vertexes[MMCVIDI.x],
                                             _vertexes[MMCVIDI.y],
                                             _vertexes[MMCVIDI.z],
                                             _vertexes[MMCVIDI.w],
                                             distCoord[last_index]);
            Friction::computeTangentBasis_EE(_vertexes[MMCVIDI.x],
                                             _vertexes[MMCVIDI.y],
                                             _vertexes[MMCVIDI.z],
                                             _vertexes[MMCVIDI.w],
                                             tanBasis[last_index]);
        }
    }
    else
    {
        int v0I = -MMCVIDI.x - 1;
        if(MMCVIDI.z < 0)
        {
            if(MMCVIDI.y >= 0)
            {
                last_index = gipc::ATOMIC_ADD(_cpNum_last, 1);
                gipc::ATOMIC_ADD(_cpNum_last + 2, 1);
                _d_PP(_vertexes[v0I], _vertexes[MMCVIDI.y], dis);
                distCoord[last_index].x = 0;
                distCoord[last_index].y = 0;
                Friction::computeTangentBasis_PP(
                    _vertexes[v0I], _vertexes[MMCVIDI.y], tanBasis[last_index]);
            }
        }
        else if(MMCVIDI.w < 0)
        {
            if(MMCVIDI.y >= 0)
            {
                last_index = gipc::ATOMIC_ADD(_cpNum_last, 1);
                gipc::ATOMIC_ADD(_cpNum_last + 3, 1);
                _d_PE(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
                Friction::computeClosestPoint_PE(_vertexes[v0I],
                                                 _vertexes[MMCVIDI.y],
                                                 _vertexes[MMCVIDI.z],
                                                 distCoord[last_index].x);
                distCoord[last_index].y = 0;
                Friction::computeTangentBasis_PE(_vertexes[v0I],
                                                 _vertexes[MMCVIDI.y],
                                                 _vertexes[MMCVIDI.z],
                                                 tanBasis[last_index]);
            }
        }
        else
        {
            last_index = gipc::ATOMIC_ADD(_cpNum_last, 1);
            gipc::ATOMIC_ADD(_cpNum_last + 4, 1);
            _d_PT(_vertexes[v0I],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            Friction::computeClosestPoint_PT(_vertexes[v0I],
                                             _vertexes[MMCVIDI.y],
                                             _vertexes[MMCVIDI.z],
                                             _vertexes[MMCVIDI.w],
                                             distCoord[last_index]);
            Friction::computeTangentBasis_PT(_vertexes[v0I],
                                             _vertexes[MMCVIDI.y],
                                             _vertexes[MMCVIDI.z],
                                             _vertexes[MMCVIDI.w],
                                             tanBasis[last_index]);
        }
    }
    if(last_index >= 0)
    {
//        double t = dis - dHat;
//        lambda_lastH[last_index] = -Kappa * 2.0 * std::sqrt(dis) * (t * std::log(dis / dHat) * -2.0 - (t * t) / dis);
#if(RANK == 1)
        double t = dis - dHat;
        lambda_lastH[last_index] =
            -Kappa * 2.0 * sqrt(dis) * (t * log(dis / dHat) * -2.0 - (t * t) / dis);
#elif(RANK == 2)
        lambda_lastH[last_index] =
            -Kappa * 2.0 * sqrt(dis)
            * (log(dis / dHat) * log(dis / dHat) * (2 * dis - 2 * dHat)
               + (2 * log(dis / dHat) * (dis - dHat) * (dis - dHat)) / dis);
#endif
        _collisionPair_last[last_index] = _collisionPair[idx];
    }
}

/// <summary>
///  host code
/// </summary>
void GIPC::ensureTimingEvents()
{
    for(cudaEvent_t& event : _frame_timing_events)
        if(!event)
            CUDA_SAFE_CALL(cudaEventCreate(&event));
    for(cudaEvent_t& event : _newton_timing_events)
        if(!event)
            CUDA_SAFE_CALL(cudaEventCreate(&event));
}

void GIPC::releaseTimingEvents()
{
    for(cudaEvent_t& event : _frame_timing_events)
    {
        if(event)
            CUDA_SAFE_CALL(cudaEventDestroy(event));
        event = nullptr;
    }
    for(cudaEvent_t& event : _newton_timing_events)
    {
        if(event)
            CUDA_SAFE_CALL(cudaEventDestroy(event));
        event = nullptr;
    }
}

void GIPC::FREE_DEVICE_MEM()
{
    releaseTimingEvents();
    _MatIndex.release();
    _collisonPairs.release();
    _ccd_collisonPairs.release();
    _cpNum.release();
    _close_cpNum.release();
    _close_gpNum.release();
    _closeConstraintID.release();
    _closeConstraintVal.release();
    _closeMConstraintID.release();
    _closeMConstraintVal.release();
    _environment_collisionPair.release();
    _gpNum.release();
    _scalar_scratch.release();
    _distance_scratch.release();
    _groundNormal.release();
    _groundOffset.release();
    _faces.release();
    _edges.release();
    _surfVerts.release();
    lambda_lastH_scalar.release();
    distCoord.release();
    tanBasis.release();
    _collisonPairs_lastH.release();
    _MatIndex_last.release();
    lambda_lastH_scalar_gd.release();
    _collisonPairs_lastH_gd.release();

    pcg_data.FREE_DEVICE_MEM();

    bvh_e.FREE_DEVICE_MEM();
    bvh_f.FREE_DEVICE_MEM();
    BH.FREE_DEVICE_MEM();
}

void GIPC::MALLOC_DEVICE_MEM()
{
    constexpr size_t dcd_capacity = 100'000;
    constexpr size_t ccd_capacity = 1'000'000;
    _MatIndex.reserve(dcd_capacity);
    _collisonPairs.reserve(dcd_capacity);
    _ccd_collisonPairs.reserve(ccd_capacity);
    _environment_collisionPair.resize(surf_vertexNum);
    _cpNum.resize(5);
    _gpNum.resize(1);
    _groundNormal.resize(5);
    _groundOffset.resize(5);
    double  h_offset[5] = {-1, -1, 1, -1, 1};
    double3 H_normal[5];  // = { make_double3(0, 1, 0);
    H_normal[0] = make_double3(0, 1, 0);
    H_normal[1] = make_double3(1, 0, 0);
    H_normal[2] = make_double3(-1, 0, 0);
    H_normal[3] = make_double3(0, 0, 1);
    H_normal[4] = make_double3(0, 0, -1);
    CUDA_SAFE_CALL(cudaMemcpy(_groundOffset, &h_offset, 5 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(_groundNormal, &H_normal, 5 * sizeof(double3), cudaMemcpyHostToDevice));


    _faces.resize(surface_Num);
    _edges.resize(edge_Num);
    _surfVerts.resize(surf_vertexNum);
    _close_cpNum.resize(1);
    _close_gpNum.resize(1);

    CUDA_SAFE_CALL(cudaMemset(_close_cpNum, 0, sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemset(_close_gpNum, 0, sizeof(uint32_t)));

    pcg_data.Malloc_DEVICE_MEM(vertexNum, tetrahedraNum);
}


void GIPC::initBVH(int* _btype)
{

    bvh_e.init(_btype,
               _vertexes,
               _rest_vertexes,
               _edges,
               edge_Num,
               surf_vertexNum);
    bvh_f.init(_btype, _vertexes, _faces, _surfVerts, surface_Num, surf_vertexNum);
}

void GIPC::init(double m_meanMass, double m_meanVolumn)
{
    SceneSize     = bvh_f.scene;
    bboxDiagSize2 = __GEIGEN__::__squaredNorm(
        __GEIGEN__::__minus(SceneSize.upper, SceneSize.lower));  //(maxConer - minConer).squaredNorm();
    dTol         = 1e-18 * bboxDiagSize2;
    minKappaCoef = 1e11;
    meanMass     = m_meanMass;
    meanVolumn   = m_meanVolumn;
    dHat         = relative_dhat * relative_dhat * bboxDiagSize2;  //__GEIGEN__::__squaredNorm(__GEIGEN__::__minus(maxConer, minConer));
    fDhat        = 1e-6 * bboxDiagSize2;
    BH.MALLOC_DEVICE_MEM_O(tetrahedraNum, surf_vertexNum, surface_Num, edge_Num, triangleNum, tri_edge_num);
}

GIPC::~GIPC()
{
    FREE_DEVICE_MEM();
}

GIPC::GIPC()
{
    IPC_dt            = 0.01;
    animation_subRate = 1.0;
    animation         = false;

    h_cpNum_last[0] = 0;
    h_cpNum_last[1] = 0;
    h_cpNum_last[2] = 0;
    h_cpNum_last[3] = 0;
    h_cpNum_last[4] = 0;
}

void GIPC::buildFrictionSets()
{
    CUDA_SAFE_CALL(cudaMemset(_cpNum, 0, 5 * sizeof(uint32_t)));
    int                numbers   = h_cpNum[0];
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    if(numbers > 0)
    {
        _calFrictionLastH_DistAndTan<<<blockNum, threadNum>>>(_vertexes,
                                                              _collisonPairs,
                                                              lambda_lastH_scalar,
                                                              distCoord,
                                                              tanBasis,
                                                              _collisonPairs_lastH,
                                                              dHat,
                                                              Kappa,
                                                              _cpNum,
                                                              h_cpNum[0]);
        
    }
    CUDA_SAFE_CALL(cudaMemcpy(h_cpNum_last, _cpNum, 5 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    numbers = h_gpNum;
    if(numbers > 0)
    {

        blockNum = (numbers + threadNum - 1) / threadNum;
        _calFrictionLastH_gd<<<blockNum, threadNum>>>(_vertexes,
                                                      _groundOffset,
                                                      _groundNormal,
                                                      _environment_collisionPair,
                                                      lambda_lastH_scalar_gd,
                                                      _collisonPairs_lastH_gd,
                                                      dHat,
                                                      Kappa,
                                                      h_gpNum);
    }
    h_gpNum_last = h_gpNum;
}


void GIPC::GroundCollisionDetect()
{
    int                numbers   = surf_vertexNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _GroundCollisionDetect<<<blockNum, threadNum>>>(
        _vertexes, _surfVerts, _groundOffset, _groundNormal, _environment_collisionPair, _gpNum, dHat, numbers);
}

void GIPC::computeSoftConstraintGradientAndHessian(double3* _gradient)
{
    int numbers = softNum;
    if(numbers < 1)
    {
        CUDA_SAFE_CALL(cudaMemcpy(&BH.DNum, _gpNum, sizeof(int), cudaMemcpyDeviceToHost));
        return;
    }
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    // offset
    _computeSoftConstraintGradientAndHessian<<<blockNum, threadNum>>>(
        _vertexes, targetVert, targetInd, _gradient, _gpNum, BH.H3x3, BH.D1Index, softMotionRate, animation_fullRate, softNum);
    CUDA_SAFE_CALL(cudaMemcpy(&BH.DNum, _gpNum, sizeof(int), cudaMemcpyDeviceToHost));
}

void GIPC::computeGroundGradientAndHessian(double3* _gradient)
{
#ifndef USE_FRICTION
    CUDA_SAFE_CALL(cudaMemset(_gpNum, 0, sizeof(uint32_t)));
#endif
    int numbers = h_gpNum;
    if(numbers < 1)
    {
        CUDA_SAFE_CALL(cudaMemcpy(&BH.DNum, _gpNum, sizeof(int), cudaMemcpyDeviceToHost));
        return;
    }
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _computeGroundGradientAndHessian<<<blockNum, threadNum>>>(_vertexes,
                                                              _groundOffset,
                                                              _groundNormal,
                                                              _environment_collisionPair,
                                                              _gradient,
                                                              _gpNum,
                                                              BH.H3x3,
                                                              BH.D1Index,
                                                              dHat,
                                                              Kappa,
                                                              numbers);
    CUDA_SAFE_CALL(cudaMemcpy(&BH.DNum, _gpNum, sizeof(int), cudaMemcpyDeviceToHost));
}

void GIPC::computeCloseGroundVal()
{
    int                numbers   = h_gpNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _computeGroundCloseVal<<<blockNum, threadNum>>>(_vertexes,
                                                    _groundOffset,
                                                    _groundNormal,
                                                    _environment_collisionPair,
                                                    dTol,
                                                    _closeConstraintID,
                                                    _closeConstraintVal,
                                                    _close_gpNum,
                                                    numbers);
}

bool GIPC::checkCloseGroundVal()
{
    int numbers = h_close_gpNum;
    if(numbers < 1)
        return false;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _scalar_scratch.resize_discard(1);
    CUDA_SAFE_CALL(cudaMemset(_scalar_scratch.data(), 0, sizeof(int)));
    _checkGroundCloseVal<<<blockNum, threadNum>>>(
        _vertexes, _groundOffset, _groundNormal, _scalar_scratch.data(), _closeConstraintID, _closeConstraintVal, numbers);
    int isChange;
    CUDA_SAFE_CALL(cudaMemcpy(
        &isChange, _scalar_scratch.data(), sizeof(int), cudaMemcpyDeviceToHost));

    return (isChange == 1);
}

// =============================================================================
// Second-stage reduction drivers
//
// Collapse the per-block partial results of a first-stage reduction kernel
// down to one scalar on the host (loop of block reductions + one DtoH copy).
// Shared by all host-side reduction routines below (behavior unchanged).
// =============================================================================

static double __reducePartialsToScalar_Sum(double* queue, int numbers)
{
    return gpuipc::DeviceReduce().sum(queue, numbers);
}

static double __reducePartialsToScalar_Max(double* queue, int numbers)
{
    return gpuipc::DeviceReduce().max(queue, numbers);
}

static double2 __reducePartialsToScalar_Max2(double2* queue, int numbers)
{
    return gpuipc::DeviceReduce().component_max(queue, numbers);
}

double2 GIPC::minMaxGroundDist()
{
    //_reduct_minGroundDist << <blockNum, threadNum >> > (_vertexes, _groundOffset, _groundNormal, _isChange, _closeConstraintID, _closeConstraintVal, numbers);

    int numbers = h_gpNum;
    if(numbers < 1)
        return make_double2(1e32, 0);
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double2) * (threadNum >> 5);

    _distance_scratch.resize_discard(numbers);
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
    _reduct_MGroundDist<<<blockNum, threadNum, sharedMsize>>>(
        _vertexes, _groundOffset, _groundNormal, _environment_collisionPair, _distance_scratch.data(), numbers);
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    double2 minMaxValue =
        __reducePartialsToScalar_Max2(_distance_scratch.data(), blockNum);
    minMaxValue.x = 1.0 / minMaxValue.x;
    return minMaxValue;
}

void GIPC::computeGroundGradient(double3* _gradient, double mKappa)
{
    int                numbers   = h_gpNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _computeGroundGradient<<<blockNum, threadNum>>>(_vertexes,
                                                    _groundOffset,
                                                    _groundNormal,
                                                    _environment_collisionPair,
                                                    _gradient,
                                                    _gpNum,
                                                    BH.H3x3,
                                                    dHat,
                                                    mKappa,
                                                    numbers);
}

void GIPC::computeSoftConstraintGradient(double3* _gradient)
{
    int numbers = softNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    // offset
    _computeSoftConstraintGradient<<<blockNum, threadNum>>>(
        _vertexes, targetVert, targetInd, _gradient, softMotionRate, animation_fullRate, softNum);
}

double GIPC::self_largestFeasibleStepSize(double slackness, double* mqueue, int numbers)
{
    //slackness = 0.9;
    //int numbers = h_cpNum[0];
    if(numbers < 1)
        return 1;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    //double* _minSteps;
    //CUDA_SAFE_CALL(cudaMalloc((void**)&_minSteps, numbers * sizeof(double)));
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
    _reduct_min_selfTimeStep_to_double<<<blockNum, threadNum, sharedMsize>>>(
        _vertexes, _ccd_collisonPairs, _moveDir, mqueue, slackness, numbers);
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    double minValue = __reducePartialsToScalar_Max(mqueue, blockNum);
    //printf("                 full ccd time step:  %f\n", 1.0 / minValue);
    //CUDA_SAFE_CALL(cudaFree(_minSteps));
    return 1.0 / minValue;
}

double GIPC::cfl_largestSpeed(double* mqueue)
{
    int                numbers   = surf_vertexNum;
    //if(numbers < 1)
    //    return ;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    /*double* _maxV;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_maxV, numbers * sizeof(double)));*/
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
    _reduct_max_cfl_to_double<<<blockNum, threadNum, sharedMsize>>>(
        _moveDir, mqueue, _surfVerts, numbers);
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    double minValue = __reducePartialsToScalar_Max(mqueue, blockNum);
    //CUDA_SAFE_CALL(cudaFree(_maxV));
    return minValue;
}

double reduction2Kappa(int type, const double3* A, const double3* B, double* _queue, int vertexNum)
{
    int                numbers   = vertexNum;
    //if(numbers < 1)
    //    return ;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    /*double* _queue;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_queue, numbers * sizeof(double)));*/
    if(type == 0)
    {
        //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
        _reduct_double3Dot_to_double<<<blockNum, threadNum, sharedMsize>>>(A, B, _queue, numbers);
    }
    else if(type == 1)
    {
        _reduct_double3Sqn_to_double<<<blockNum, threadNum, sharedMsize>>>(A, _queue, numbers);
    }
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    double dotValue = __reducePartialsToScalar_Sum(_queue, blockNum);
    //CUDA_SAFE_CALL(cudaFree(_queue));
    return dotValue;
}

double GIPC::ground_largestFeasibleStepSize(double slackness, double* mqueue)
{

    int                numbers   = surf_vertexNum;
    if(numbers < 1)
        return 1;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    //double* _minSteps;
    //CUDA_SAFE_CALL(cudaMalloc((void**)&_minSteps, numbers * sizeof(double)));

    //if (h_cpNum[0] > 0) {
    //    double3* mvd = new double3[vertexNum];
    //    cudaMemcpy(mvd, _moveDir, sizeof(double3) * vertexNum, cudaMemcpyDeviceToHost);
    //    for (int i = 0;i < vertexNum;i++) {
    //        printf("%f  %f  %f\n", mvd[i].x, mvd[i].y, mvd[i].z);
    //    }
    //    delete[] mvd;
    //}
    _reduct_min_groundTimeStep_to_double<<<blockNum, threadNum, sharedMsize>>>(
        _vertexes, _surfVerts, _groundOffset, _groundNormal, _moveDir, mqueue, slackness, numbers);


    double minValue = __reducePartialsToScalar_Max(mqueue, blockNum);
    //CUDA_SAFE_CALL(cudaFree(_minSteps));
    return 1.0 / minValue;
}

double GIPC::InjectiveStepSize(double slackness, double errorRate, double* mqueue, uint4* tets)
{

    int                numbers   = tetrahedraNum;
    if(numbers < 1)
        return 1;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    _reduct_min_InjectiveTimeStep_to_double<<<blockNum, threadNum, sharedMsize>>>(
        _vertexes, tets, _moveDir, mqueue, slackness, errorRate, numbers);


    double minValue = __reducePartialsToScalar_Max(mqueue, blockNum);
    //printf("Injective Time step:   %f\n", 1.0 / minValue);
    //if (1.0 / minValue < 1) {
    //    system("pause");
    //}
    //CUDA_SAFE_CALL(cudaFree(_minSteps));
    return 1.0 / minValue;
}

void GIPC::buildCP()
{
    for(;;)
    {
        const size_t common_capacity = std::min({_collisonPairs.capacity(),
                                                 _ccd_collisonPairs.capacity(),
                                                 _MatIndex.capacity()});
        if(common_capacity == 0
           || common_capacity > std::numeric_limits<uint32_t>::max())
        {
            std::cerr << "Invalid DCD pair capacity: " << common_capacity << std::endl;
            std::abort();
        }
        const uint32_t pair_capacity = static_cast<uint32_t>(common_capacity);

        _collisonPairs.resize(_collisonPairs.capacity());
        _ccd_collisonPairs.resize(_ccd_collisonPairs.capacity());
        _MatIndex.resize(_MatIndex.capacity());

        CUDA_SAFE_CALL(cudaMemset(_cpNum, 0, 5 * sizeof(uint32_t)));
        CUDA_SAFE_CALL(cudaMemset(_gpNum, 0, sizeof(uint32_t)));
        bvh_f.SelfCollitionDetect(dHat,
                                  _collisonPairs.data(),
                                  _ccd_collisonPairs.data(),
                                  _cpNum.data(),
                                  _MatIndex.data(),
                                  pair_capacity);
        bvh_e.SelfCollitionDetect(dHat,
                                  _collisonPairs.data(),
                                  _ccd_collisonPairs.data(),
                                  _cpNum.data(),
                                  _MatIndex.data(),
                                  pair_capacity);
        GroundCollisionDetect();
        CUDA_SAFE_CALL(cudaMemcpy(
            &h_cpNum, _cpNum.data(), 5 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(
            &h_gpNum, _gpNum.data(), sizeof(uint32_t), cudaMemcpyDeviceToHost));

        if(h_cpNum[0] <= pair_capacity)
        {
            const uint64_t typed_count = static_cast<uint64_t>(h_cpNum[2])
                                         + static_cast<uint64_t>(h_cpNum[3])
                                         + static_cast<uint64_t>(h_cpNum[4]);
            if(typed_count != h_cpNum[0])
            {
                std::cerr << "Collision subtype count mismatch: total=" << h_cpNum[0]
                          << ", typed=" << typed_count << std::endl;
                std::abort();
            }
            _collisonPairs.resize(h_cpNum[0]);
            _ccd_collisonPairs.resize(h_cpNum[0]);
            _MatIndex.resize(h_cpNum[0]);
            pcg_data.MP.updateCollisionPairs(_collisonPairs.data());
            break;
        }

        std::cout << "Growing DCD pair capacity from " << pair_capacity << " for "
                  << h_cpNum[0] << " detected pairs." << std::endl;
        _collisonPairs.resize_discard(h_cpNum[0]);
        _ccd_collisonPairs.resize_discard(h_cpNum[0]);
        _MatIndex.resize_discard(h_cpNum[0]);
    }
}

void GIPC::buildFullCP(const double& alpha)
{
    for(;;)
    {
        const size_t capacity = _ccd_collisonPairs.capacity();
        if(capacity == 0 || capacity > std::numeric_limits<uint32_t>::max())
        {
            std::cerr << "Invalid CCD pair capacity: " << capacity << std::endl;
            std::abort();
        }
        const uint32_t pair_capacity = static_cast<uint32_t>(capacity);
        _ccd_collisonPairs.resize(capacity);

        CUDA_SAFE_CALL(cudaMemset(_cpNum, 0, sizeof(uint32_t)));
        bvh_f.SelfCollitionFullDetect(dHat,
                                      _moveDir,
                                      alpha,
                                      _ccd_collisonPairs.data(),
                                      _cpNum.data(),
                                      pair_capacity);
        bvh_e.SelfCollitionFullDetect(dHat,
                                      _moveDir,
                                      alpha,
                                      _ccd_collisonPairs.data(),
                                      _cpNum.data(),
                                      pair_capacity);
        CUDA_SAFE_CALL(cudaMemcpy(
            &h_ccd_cpNum, _cpNum.data(), sizeof(uint32_t), cudaMemcpyDeviceToHost));

        if(h_ccd_cpNum <= pair_capacity)
        {
            _ccd_collisonPairs.resize(h_ccd_cpNum);
            break;
        }

        std::cout << "Growing CCD pair capacity from " << pair_capacity << " for "
                  << h_ccd_cpNum << " detected pairs." << std::endl;
        _ccd_collisonPairs.resize_discard(h_ccd_cpNum);
    }
}


void GIPC::buildBVH()
{
    bvh_f.Construct();
    bvh_e.Construct();
}

AABB* GIPC::calcuMaxSceneSize()
{
    return bvh_f.getSceneSize();
}

void GIPC::buildBVH_FULLCCD(const double& alpha)
{
    bvh_f.ConstructFullCCD(_moveDir, alpha);
    bvh_e.ConstructFullCCD(_moveDir, alpha);
}

void GIPC::calBarrierGradientAndHessian(double3* _gradient, double mKappa)
{
    int numbers = h_cpNum[0];
    if(numbers < 1)
        return;
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;


    _calBarrierGradientAndHessian<<<blockNum, threadNum>>>(_vertexes,
                                                           _rest_vertexes,
                                                           _collisonPairs,
                                                           _gradient,
                                                           BH.H12x12,
                                                           BH.H9x9,
                                                           BH.H6x6,
                                                           BH.D4Index,
                                                           BH.D3Index,
                                                           BH.D2Index,
                                                           _cpNum,
                                                           _MatIndex,
                                                           dHat,
                                                           mKappa,
                                                           numbers);
}


void GIPC::calFrictionHessian(device_TetraData& TetMesh)
{
    int numbers = h_cpNum_last[0];
    //if (numbers < 1) return;
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    if(numbers > 0)
    {
        _calFrictionHessian<<<blockNum, threadNum>>>(_vertexes,
                                                     TetMesh.o_vertexes,
                                                     _collisonPairs_lastH,
                                                     BH.H12x12,
                                                     BH.H9x9,
                                                     BH.H6x6,
                                                     BH.D4Index,
                                                     BH.D3Index,
                                                     BH.D2Index,
                                                     _cpNum,
                                                     numbers,
                                                     IPC_dt,
                                                     distCoord,
                                                     tanBasis,
                                                     fDhat * IPC_dt * IPC_dt,
                                                     lambda_lastH_scalar,
                                                     frictionRate,
                                                     h_cpNum[4],
                                                     h_cpNum[3],
                                                     h_cpNum[2]);
    }

    numbers = h_gpNum_last;
    CUDA_SAFE_CALL(cudaMemcpy(_gpNum, &h_gpNum_last, sizeof(uint32_t), cudaMemcpyHostToDevice));
    if(numbers < 1)
        return;
    
    blockNum = (numbers + threadNum - 1) / threadNum;

    _calFrictionHessian_gd<<<blockNum, threadNum>>>(_vertexes,
                                                    TetMesh.o_vertexes,
                                                    _groundNormal,
                                                    _collisonPairs_lastH_gd,
                                                    BH.H3x3,
                                                    BH.D1Index,
                                                    numbers,
                                                    IPC_dt,
                                                    fDhat * IPC_dt * IPC_dt,
                                                    lambda_lastH_scalar_gd,
                                                    frictionRate);
}

void GIPC::computeSelfCloseVal()
{
    int                numbers   = h_cpNum[0];
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _calSelfCloseVal<<<blockNum, threadNum>>>(
        _vertexes, _collisonPairs, _closeMConstraintID, _closeMConstraintVal, _close_cpNum, dTol, numbers);
}

bool GIPC::checkSelfCloseVal()
{
    int numbers = h_close_cpNum;
    if(numbers < 1)
        return false;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _scalar_scratch.resize_discard(1);
    CUDA_SAFE_CALL(cudaMemset(_scalar_scratch.data(), 0, sizeof(int)));
    _checkSelfCloseVal<<<blockNum, threadNum>>>(
        _vertexes, _scalar_scratch.data(), _closeMConstraintID, _closeMConstraintVal, numbers);
    int isChange;
    CUDA_SAFE_CALL(cudaMemcpy(
        &isChange, _scalar_scratch.data(), sizeof(int), cudaMemcpyDeviceToHost));

    return (isChange == 1);
}

double2 GIPC::minMaxSelfDist()
{
    int numbers = h_cpNum[0];
    if(numbers < 1)
        return make_double2(1e32, 0);
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double2) * (threadNum >> 5);

    _distance_scratch.resize_discard(numbers);
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
    _reduct_MSelfDist<<<blockNum, threadNum, sharedMsize>>>(
        _vertexes, _collisonPairs, _distance_scratch.data(), numbers);
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    double2 minValue = __reducePartialsToScalar_Max2(_distance_scratch.data(), blockNum);
    minValue.x = 1.0 / minValue.x;
    return minValue;
}

void GIPC::calBarrierGradient(double3* _gradient, double mKappa)
{
    int numbers = h_cpNum[0];
    if(numbers < 1)
        return;
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    _calBarrierGradient<<<blockNum, threadNum>>>(
        _vertexes, _rest_vertexes, _collisonPairs, _gradient, dHat, mKappa, numbers);
}

void GIPC::calFrictionGradient(double3* _gradient, device_TetraData& TetMesh)
{
    int numbers = h_cpNum_last[0];
    const unsigned int threadNum = 256;
    int blockNum = 0;
    if(numbers > 0)
    {
        blockNum  = (numbers + threadNum - 1) / threadNum;
        _calFrictionGradient<<<blockNum, threadNum>>>(_vertexes,
                                                      TetMesh.o_vertexes,
                                                      _collisonPairs_lastH,
                                                      _gradient,
                                                      numbers,
                                                      IPC_dt,
                                                      distCoord,
                                                      tanBasis,
                                                      fDhat * IPC_dt * IPC_dt,
                                                      lambda_lastH_scalar,
                                                      frictionRate);
    }
    numbers = h_gpNum_last;
    if(numbers < 1)
        return;
    blockNum = (numbers + threadNum - 1) / threadNum;

    _calFrictionGradient_gd<<<blockNum, threadNum>>>(_vertexes,
                                                     TetMesh.o_vertexes,
                                                     _groundNormal,
                                                     _collisonPairs_lastH_gd,
                                                     _gradient,
                                                     numbers,
                                                     IPC_dt,
                                                     fDhat * IPC_dt * IPC_dt,
                                                     lambda_lastH_scalar_gd,
                                                     frictionRate);
}


void calKineticGradient(double3* _vertexes, double3* _xTilta, double3* _gradient, double* _masses, int numbers)
{
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calKineticGradient<<<blockNum, threadNum>>>(_vertexes, _xTilta, _gradient, _masses, numbers);
}

void calculate_fem_gradient_hessian(__GEIGEN__::Matrix3x3d*   DmInverses,
                                    const double3*            vertexes,
                                    const uint4*              tetrahedras,
                                    __GEIGEN__::Matrix12x12d* Hessians,
                                    const uint32_t&           offset,
                                    const double*             volume,
                                    double3*                  gradient,
                                    int                       tetrahedraNum,
                                    double                    lenRate,
                                    double                    volRate,
                                    double                    IPC_dt)
{
    int numbers = tetrahedraNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calculate_fem_gradient_hessian<<<blockNum, threadNum>>>(
        DmInverses, vertexes, tetrahedras, Hessians, offset, volume, gradient, tetrahedraNum, lenRate, volRate, IPC_dt);
}

void calculate_triangle_fem_gradient_hessian(__GEIGEN__::Matrix2x2d* triDmInverses,
                                             const double3*          vertexes,
                                             const uint3*            triangles,
                                             __GEIGEN__::Matrix9x9d* Hessians,
                                             const uint32_t&         offset,
                                             const double*           area,
                                             double3*                gradient,
                                             int    triangleNum,
                                             double stretchStiff,
                                             double shearStiff,
                                             double IPC_dt)
{
    int numbers = triangleNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calculate_triangle_fem_gradient_hessian<<<blockNum, threadNum>>>(
        triDmInverses, vertexes, triangles, Hessians, offset, area, gradient, triangleNum, stretchStiff, shearStiff, IPC_dt);
}

void calculate_bending_gradient_hessian(const double3* vertexes,
                                        const double3* rest_vertexes,
                                        const uint2*   edges,
                                        const uint2*   edges_adj_vertex,
                                        __GEIGEN__::Matrix12x12d* Hessians,
                                        uint4*                    Indices,
                                        const uint32_t&           offset,
                                        double3*                  gradient,
                                        int                       edgeNum,
                                        double                    bendStiff,
                                        double                    IPC_dt)
{
    int numbers = edgeNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calculate_bending_gradient_hessian<<<blockNum, threadNum>>>(
        vertexes, rest_vertexes, edges, edges_adj_vertex, Hessians, Indices, offset, gradient, edgeNum, bendStiff, IPC_dt);
}

void calculate_fem_gradient(__GEIGEN__::Matrix3x3d* DmInverses,
                            const double3*          vertexes,
                            const uint4*            tetrahedras,
                            const double*           volume,
                            double3*                gradient,
                            int                     tetrahedraNum,
                            double                  lenRate,
                            double                  volRate,
                            double                  dt)
{
    int                numbers   = tetrahedraNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calculate_fem_gradient<<<blockNum, threadNum>>>(
        DmInverses, vertexes, tetrahedras, volume, gradient, tetrahedraNum, lenRate, volRate, dt);
}

void calculate_triangle_fem_gradient(__GEIGEN__::Matrix2x2d* triDmInverses,
                                     const double3*          vertexes,
                                     const uint3*            triangles,
                                     const double*           area,
                                     double3*                gradient,
                                     int                     triangleNum,
                                     double                  stretchStiff,
                                     double                  shearStiff,
                                     double                  IPC_dt)
{
    int numbers = triangleNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calculate_triangle_fem_gradient<<<blockNum, threadNum>>>(
        triDmInverses, vertexes, triangles, area, gradient, triangleNum, stretchStiff, shearStiff, IPC_dt);
}

double calcMinMovement(const double3* _moveDir, double* _queue, const int& number)
{

    int                numbers   = number;
    if(numbers < 1)
        return 0;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    /*double* _tempMinMovement;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_tempMinMovement, numbers * sizeof(double)));*/
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));

    _reduct_max_double3_to_double<<<blockNum, threadNum, sharedMsize>>>(_moveDir, _queue, numbers);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

    double minValue = __reducePartialsToScalar_Max(_queue, blockNum);
    //CUDA_SAFE_CALL(cudaFree(_tempMinMovement));
    return minValue;
}

void stepForward(double3* _vertexes,
                 double3* _vertexesTemp,
                 double3* _moveDir,
                 int*     bType,
                 double   alpha,
                 bool     moveBoundary,
                 int      numbers)
{
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _stepForward<<<blockNum, threadNum>>>(
        _vertexes, _vertexesTemp, _moveDir, bType, alpha, moveBoundary, numbers);
}


void updateSurfaces(uint32_t* sortIndex, uint3* _faces, const int& offset_num, const int& numbers)
{
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _updateSurfaces<<<blockNum, threadNum>>>(sortIndex, _faces, offset_num, numbers);
}

void updateSurfaceEdges(uint32_t* sortIndex, uint2* _edges, const int& offset_num, const int& numbers)
{
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _updateEdges<<<blockNum, threadNum>>>(sortIndex, _edges, offset_num, numbers);
}

void updateTriEdges_adjVerts(uint32_t*  sortIndex,
                             uint2*     _tri_edges,
                             uint2*     _adj_verts,
                             const int& offset_num,
                             const int& numbers)
{
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _updateTriEdges_adjVerts<<<blockNum, threadNum>>>(
        sortIndex, _tri_edges, _adj_verts, offset_num, numbers);
}


void updateSurfaceVerts(uint32_t* sortIndex, uint32_t* _sVerts, const int& offset_num, const int& numbers)
{
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _updateSurfVerts<<<blockNum, threadNum>>>(sortIndex, _sVerts, offset_num, numbers);
}

void updateNeighborInfo(unsigned int*   _neighborList,
                        unsigned int*   d_neighborListInit,
                        unsigned int*   _neighborNum,
                        unsigned int*   _neighborNumInit,
                        unsigned int*   _neighborStart,
                        unsigned int*   _neighborStartTemp,
                        const uint32_t* sortIndex,
                        const uint32_t* sortMapVertIndex,
                        const int&      numbers,
                        const int&      neighborListSize)
{
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _updateNeighborNum<<<blockNum, threadNum>>>(_neighborNumInit, _neighborNum, sortIndex, numbers);
    gpuipc::DeviceScan().exclusive_sum(_neighborNum, _neighborStartTemp, numbers);
    _updateNeighborList<<<blockNum, threadNum>>>(d_neighborListInit,
                                                 _neighborList,
                                                 _neighborNum,
                                                 _neighborStart,
                                                 _neighborStartTemp,
                                                 sortIndex,
                                                 sortMapVertIndex,
                                                 numbers);
    CUDA_SAFE_CALL(cudaMemcpy(d_neighborListInit,
                              _neighborList,
                              neighborListSize * sizeof(unsigned int),
                              cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(_neighborStart,
                              _neighborStartTemp,
                              numbers * sizeof(unsigned int),
                              cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(
        _neighborNumInit, _neighborNum, numbers * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
}

void calcTetMChash(uint64_t*         _MChash,
                   const double3*    _vertexes,
                   uint4*            tets,
                   const AABB* _MaxBv,
                   const uint32_t*   sortMapVertIndex,
                   int               number)
{
    int                numbers   = number;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calcTetMChash<<<blockNum, threadNum>>>(
        _MChash, _vertexes, tets, _MaxBv, sortMapVertIndex, number);
}

void updateTopology(uint4* tets, uint3* tris, const uint32_t* sortMapVertIndex, int traNumber, int triNumber)
{
    int                numbers   = __m_max(traNumber, triNumber);
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _updateTopology<<<blockNum, threadNum>>>(tets, tris, sortMapVertIndex, traNumber, triNumber);
}

void updateVertexes(double3*                      o_vertexes,
                    const double3*                _vertexes,
                    double*                       tempM,
                    const double*                 mass,
                    __GEIGEN__::Matrix3x3d*       tempCons,
                    int*                          tempBtype,
                    const __GEIGEN__::Matrix3x3d* cons,
                    const int*                    bType,
                    const uint32_t*               sortIndex,
                    uint32_t*                     sortMapIndex,
                    int                           number)
{
    int                numbers   = number;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _updateVertexes<<<blockNum, threadNum>>>(
        o_vertexes, _vertexes, tempM, mass, tempCons, tempBtype, cons, bType, sortIndex, sortMapIndex, numbers);
}

void updateTetrahedras(uint4*                        o_tetrahedras,
                       uint4*                        tetrahedras,
                       double*                       tempV,
                       const double*                 volum,
                       __GEIGEN__::Matrix3x3d*       tempDmInverse,
                       const __GEIGEN__::Matrix3x3d* dmInverse,
                       const uint32_t*               sortTetIndex,
                       const uint32_t*               sortMapVertIndex,
                       int                           number)
{
    int                numbers   = number;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _updateTetrahedras<<<blockNum, threadNum>>>(
        o_tetrahedras, tetrahedras, tempV, volum, tempDmInverse, dmInverse, sortTetIndex, sortMapVertIndex, number);
}

void calcVertMChash(uint64_t* _MChash, const double3* _vertexes, const AABB* _MaxBv, int number)
{
    int                numbers   = number;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calcVertMChash<<<blockNum, threadNum>>>(_MChash, _vertexes, _MaxBv, number);
}

void sortGeometry(device_TetraData& TetMesh,
                  const AABB*       _MaxBv,
                  const int&        vertex_num,
                  const int&        tetradedra_num,
                  const int&        triangle_num)
{
    calcVertMChash(TetMesh.MChash, TetMesh.vertexes, _MaxBv, vertex_num);
    fill_sort_indices<<<(vertex_num + 255) / 256, 256>>>(TetMesh.sortIndex.data(), vertex_num);
    GPUIPC_CHECK_CUDA(cudaGetLastError());
    gpuipc::DeviceRadixSort().sort_pairs(TetMesh.MChash.data(),
                                         TetMesh.sortMChash.data(),
                                         TetMesh.sortIndex.data(),
                                         TetMesh.sortIndexOutput.data(),
                                         vertex_num);
    std::swap(TetMesh.MChash, TetMesh.sortMChash);
    std::swap(TetMesh.sortIndex, TetMesh.sortIndexOutput);
    updateVertexes(TetMesh.o_vertexes,
                   TetMesh.vertexes,
                   TetMesh.tempDouble,
                   TetMesh.masses,
                   TetMesh.tempMat3x3,
                   TetMesh.tempBoundaryType,
                   TetMesh.Constraints,
                   TetMesh.BoundaryType,
                   TetMesh.sortIndex,
                   TetMesh.sortMapVertIndex,
                   vertex_num);
    CUDA_SAFE_CALL(cudaMemcpy(TetMesh.vertexes,
                              TetMesh.o_vertexes,
                              vertex_num * sizeof(double3),
                              cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(
        TetMesh.masses, TetMesh.tempDouble, vertex_num * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(TetMesh.Constraints,
                              TetMesh.tempMat3x3,
                              vertex_num * sizeof(__GEIGEN__::Matrix3x3d),
                              cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(TetMesh.BoundaryType,
                              TetMesh.tempBoundaryType,
                              vertex_num * sizeof(int),
                              cudaMemcpyDeviceToDevice));

    updateTopology(TetMesh.tetrahedras, TetMesh.triangles, TetMesh.sortMapVertIndex, tetradedra_num, triangle_num);
    //calcTetMChash(TetMesh.MChash, TetMesh.vertexes, TetMesh.tetrahedras, _MaxBv, TetMesh.sortMapVertIndex, tetradedra_num);
    //thrust::sequence(thrust::device_ptr<uint32_t>(TetMesh.sortIndex), thrust::device_ptr<uint32_t>(TetMesh.sortIndex) + tetradedra_num);
    //thrust::sort_by_key(thrust::device_ptr<uint64_t>(TetMesh.MChash), thrust::device_ptr<uint64_t>(TetMesh.MChash) + tetradedra_num, thrust::device_ptr<uint32_t>(TetMesh.sortIndex));
    //updateTetrahedras(TetMesh.tempTetrahedras, TetMesh.tetrahedras, TetMesh.tempDouble, TetMesh.volum, TetMesh.tempMat3x3, TetMesh.DmInverses, TetMesh.sortIndex, TetMesh.sortMapVertIndex, tetradedra_num);
    //CUDA_SAFE_CALL(cudaMemcpy(TetMesh.tetrahedras, TetMesh.tempTetrahedras, tetradedra_num * sizeof(uint4), cudaMemcpyDeviceToDevice));
    //CUDA_SAFE_CALL(cudaMemcpy(TetMesh.volum, TetMesh.tempDouble, tetradedra_num * sizeof(double), cudaMemcpyDeviceToDevice));
    //CUDA_SAFE_CALL(cudaMemcpy(TetMesh.DmInverses, TetMesh.tempMat3x3, tetradedra_num * sizeof(__GEIGEN__::Matrix3x3d), cudaMemcpyDeviceToDevice));
}

////////////////////////TO DO LATER/////////////////////////////////////////


void compute_H_b(double d, double dHat, double& H)
{
    double t = d - dHat;
    H = (std::log(d / dHat) * -2.0 - t * 4.0 / d) + 1.0 / (d * d) * (t * t);
}

void GIPC::suggestKappa(double& kappa)
{
    double H_b;
    //double bboxDiagSize2 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(bvh_f.scene.upper, bvh_f.scene.lower));
    compute_H_b(1.0e-16 * bboxDiagSize2, dHat, H_b);
    if(meanMass == 0.0)
    {
        kappa = minKappaCoef / (4.0e-16 * bboxDiagSize2 * H_b);
    }
    else
    {
        kappa = minKappaCoef * meanMass / (4.0e-16 * bboxDiagSize2 * H_b);
    }
    //    printf("bboxDiagSize2: %f\n", bboxDiagSize2);
    //    printf("H_b: %f\n", H_b);
    //    printf("sug Kappa: %f\n", kappa);
}

void GIPC::upperBoundKappa(double& kappa)
{
    double H_b;
    //double bboxDiagSize2 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(bvh_f.scene.upper, bvh_f.scene.lower));//(maxConer - minConer).squaredNorm();
    compute_H_b(1.0e-16 * bboxDiagSize2, dHat, H_b);
    double kappaMax = 100 * minKappaCoef * meanMass / (4.0e-16 * bboxDiagSize2 * H_b);
    //printf("max Kappa: %f\n", kappaMax);
    if(meanMass == 0.0)
    {
        kappaMax = 100 * minKappaCoef / (4.0e-16 * bboxDiagSize2 * H_b);
    }

    if(kappa > kappaMax)
    {
        kappa = kappaMax;
    }
}


void GIPC::initKappa(device_TetraData& TetMesh)
{
    if(h_cpNum[0] > 0)
    {
        double3* _GE = TetMesh.fb;
        double3* _gc = TetMesh.temp_double3Mem;
        //CUDA_SAFE_CALL(cudaMalloc((void**)&_gc, vertexNum * sizeof(double3)));
        //CUDA_SAFE_CALL(cudaMalloc((void**)&_GE, vertexNum * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMemset(_gc, 0, vertexNum * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMemset(_GE, 0, vertexNum * sizeof(double3)));
        calKineticGradient(TetMesh.vertexes, TetMesh.xTilta, _GE, TetMesh.masses, vertexNum);
        calculate_fem_gradient(TetMesh.DmInverses,
                               TetMesh.vertexes,
                               TetMesh.tetrahedras,
                               TetMesh.volum,
                               _GE,
                               tetrahedraNum,
                               lengthRate,
                               volumeRate,
                               IPC_dt);
        //calculate_triangle_fem_gradient(TetMesh.triDmInverses, TetMesh.vertexes, TetMesh.triangles, TetMesh.area, _GE, triangleNum, stretchStiff, shearStiff, IPC_dt);
        computeSoftConstraintGradient(_GE);
        computeGroundGradient(_gc, 1);
        calBarrierGradient(_gc, 1);
        double gsum = reduction2Kappa(0, _gc, _GE, pcg_data.squeue, vertexNum);
        double gsnorm = reduction2Kappa(1, _gc, _GE, pcg_data.squeue, vertexNum);
        //CUDA_SAFE_CALL(cudaFree(_gc));
        //CUDA_SAFE_CALL(cudaFree(_GE));
        double minKappa = -gsum / gsnorm;
        if(minKappa > 0.0)
        {
            Kappa = minKappa;
        }
        suggestKappa(minKappa);
        if(Kappa < minKappa)
        {
            Kappa = minKappa;
        }
        upperBoundKappa(Kappa);
    }

    //printf("Kappa ====== %f\n", Kappa);
}


float GIPC::computeGradientAndHessian(device_TetraData& TetMesh)
{
    calKineticGradient(TetMesh.vertexes, TetMesh.xTilta, TetMesh.fb, TetMesh.masses, vertexNum);
    CUDA_SAFE_CALL(cudaMemset(_cpNum, 0, 5 * sizeof(uint32_t)));
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //calBarrierHessian();
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

    calBarrierGradientAndHessian(TetMesh.fb, Kappa);

    float time00 = 0;

    //calBarrierGradient(TetMesh.fb, Kappa);
#ifdef USE_FRICTION
    calFrictionGradient(TetMesh.fb, TetMesh);
    calFrictionHessian(TetMesh);
#endif

    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    calculate_fem_gradient_hessian(TetMesh.DmInverses,
                                   TetMesh.vertexes,
                                   TetMesh.tetrahedras,
                                   BH.H12x12,
                                   h_cpNum[4] + h_cpNum_last[4],
                                   TetMesh.volum,
                                   TetMesh.fb,
                                   tetrahedraNum,
                                   lengthRate,
                                   volumeRate,
                                   IPC_dt);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(BH.D4Index + h_cpNum[4] + h_cpNum_last[4],
                              TetMesh.tetrahedras,
                              tetrahedraNum * sizeof(uint4),
                              cudaMemcpyDeviceToDevice));

    calculate_bending_gradient_hessian(TetMesh.vertexes,
                                       TetMesh.rest_vertexes,
                                       TetMesh.tri_edges,
                                       TetMesh.tri_edge_adj_vertex,
                                       BH.H12x12,
                                       BH.D4Index,
                                       h_cpNum[4] + h_cpNum_last[4] + tetrahedraNum,
                                       TetMesh.fb,
                                       tri_edge_num,
                                       bendStiff,
                                       IPC_dt);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

    calculate_triangle_fem_gradient_hessian(TetMesh.triDmInverses,
                                            TetMesh.vertexes,
                                            TetMesh.triangles,
                                            BH.H9x9,
                                            h_cpNum[3] + h_cpNum_last[3],
                                            TetMesh.area,
                                            TetMesh.fb,
                                            triangleNum,
                                            stretchStiff,
                                            shearStiff,
                                            IPC_dt);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(BH.D3Index + h_cpNum[3] + h_cpNum_last[3],
                              TetMesh.triangles,
                              triangleNum * sizeof(uint3),
                              cudaMemcpyDeviceToDevice));


    computeGroundGradientAndHessian(TetMesh.fb);
    computeSoftConstraintGradientAndHessian(TetMesh.fb);

    return time00;
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
}


double GIPC::Energy_Add_Reduction_Algorithm(int type, device_TetraData& TetMesh)
{
    int numbers = tetrahedraNum;

    if(type == 0 || type == 3)
    {
        numbers = vertexNum;
    }
    else if(type == 2)
    {
        numbers = h_cpNum[0];
    }
    else if(type == 4)
    {
        numbers = h_gpNum;
    }
    else if(type == 5)
    {
        numbers = h_cpNum_last[0];
    }
    else if(type == 6)
    {
        numbers = h_gpNum_last;
    }
    else if(type == 7 || type == 1)
    {
        numbers = tetrahedraNum;
    }
    else if(type == 8)
    {
        numbers = triangleNum;
    }
    else if(type == 9)
    {
        numbers = softNum;
    }
    else if(type == 10)
    {
        numbers = tri_edge_num;
    }
    if(numbers == 0)
        return 0;
    double* queue = pcg_data.squeue;
    //CUDA_SAFE_CALL(cudaMalloc((void**)&queue, numbers * sizeof(double)));*/

    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);
    switch(type)
    {
        case 0:
            _getKineticEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                TetMesh.vertexes, TetMesh.xTilta, queue, TetMesh.masses, numbers);
            break;
        case 1:
            _getFEMEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue,
                TetMesh.vertexes,
                TetMesh.tetrahedras,
                TetMesh.DmInverses,
                TetMesh.volum,
                numbers,
                lengthRate,
                volumeRate);
            break;
        case 2:
            _getBarrierEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, TetMesh.rest_vertexes, _collisonPairs, Kappa, dHat, numbers);
            break;
        case 3:
            _getDeltaEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.fb, _moveDir, numbers);
            break;
        case 4:
            _computeGroundEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, _groundOffset, _groundNormal, _environment_collisionPair, dHat, Kappa, numbers);
            break;
        case 5:
            _getFrictionEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue,
                TetMesh.vertexes,
                TetMesh.o_vertexes,
                _collisonPairs_lastH,
                numbers,
                IPC_dt,
                distCoord,
                tanBasis,
                lambda_lastH_scalar,
                fDhat * IPC_dt * IPC_dt,
                sqrt(fDhat) * IPC_dt);
            break;
        case 6:
            _getFrictionEnergy_gd_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue,
                TetMesh.vertexes,
                TetMesh.o_vertexes,
                _groundNormal,
                _collisonPairs_lastH_gd,
                numbers,
                IPC_dt,
                lambda_lastH_scalar_gd,
                sqrt(fDhat) * IPC_dt);
            break;
        case 7:
            _getRestStableNHKEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.volum, numbers, lengthRate, volumeRate);
            break;
        case 8:
            _get_triangleFEMEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue,
                TetMesh.vertexes,
                TetMesh.triangles,
                TetMesh.triDmInverses,
                TetMesh.area,
                numbers,
                stretchStiff,
                shearStiff);
            break;
        case 9:
            _computeSoftConstraintEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, TetMesh.targetVert, TetMesh.targetIndex, softMotionRate, animation_fullRate, numbers);
            break;
        case 10:
            _getBendingEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue,
                TetMesh.vertexes,
                TetMesh.rest_vertexes,
                TetMesh.tri_edges,
                TetMesh.tri_edge_adj_vertex,
                numbers,
                bendStiff);
            break;
    }
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    double result = __reducePartialsToScalar_Sum(queue, blockNum);
    //CUDA_SAFE_CALL(cudaFree(queue));
    return result;
}


double GIPC::computeEnergy(device_TetraData& TetMesh)
{
    double Energy = Energy_Add_Reduction_Algorithm(0, TetMesh);

    Energy += IPC_dt * IPC_dt * Energy_Add_Reduction_Algorithm(1, TetMesh);

    Energy += IPC_dt * IPC_dt * Energy_Add_Reduction_Algorithm(8, TetMesh);

    Energy += IPC_dt * IPC_dt * Energy_Add_Reduction_Algorithm(10, TetMesh);

    Energy += Energy_Add_Reduction_Algorithm(9, TetMesh);

    Energy += Energy_Add_Reduction_Algorithm(2, TetMesh);

    Energy += Kappa * Energy_Add_Reduction_Algorithm(4, TetMesh);

#ifdef USE_FRICTION
    Energy += frictionRate * Energy_Add_Reduction_Algorithm(5, TetMesh);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    Energy += frictionRate * Energy_Add_Reduction_Algorithm(6, TetMesh);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
#endif
    return Energy;
}

int GIPC::calculateMovingDirection(device_TetraData& TetMesh, int cpNum, int preconditioner_type)
{
    if(!preconditioner_type)
    {
        return PCG_Process(
            &TetMesh, &pcg_data, BH, _moveDir, vertexNum, tetrahedraNum, IPC_dt, meanVolumn, pcg_threshold);
    }
    else if(preconditioner_type == 1)
    {
        int cgCount = MASPCG_Process(
            &TetMesh, &pcg_data, BH, _moveDir, vertexNum, tetrahedraNum, IPC_dt, meanVolumn, cpNum, pcg_threshold);
        if(cgCount == 3000)
        {
            printf("MASPCG fail, turn to PCG\n");
            cgCount = PCG_Process(
                &TetMesh, &pcg_data, BH, _moveDir, vertexNum, tetrahedraNum, IPC_dt, meanVolumn, pcg_threshold);
            printf("PCG finish:  %d\n", cgCount);
        }

        return cgCount;
    }

    std::cerr << "Unsupported preconditioner type: " << preconditioner_type << std::endl;
    std::abort();
}


bool edgeTriIntersectionQuery(const int*     _btype,
                              const double3* _vertexes,
                              const uint2*   _edges,
                              const uint3*   _faces,
                              const AABB*    _edge_bvs,
                              const Node*    _edge_nodes,
                              int*           is_intersect,
                              double         dHat,
                              int            edge_count,
                              int            number)
{
    int                numbers   = number;
    if(numbers < 1 || edge_count < 1)
        return false;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    CUDA_SAFE_CALL(cudaMemset(is_intersect, 0, sizeof(int)));

    _edgeTriIntersectionQuery<<<blockNum, threadNum>>>(
        _btype,
        _vertexes,
        _edges,
        _faces,
        _edge_bvs,
        _edge_nodes,
        is_intersect,
        dHat,
        edge_count,
        numbers);

    int h_isITST;
    CUDA_SAFE_CALL(
        cudaMemcpy(&h_isITST, is_intersect, sizeof(int), cudaMemcpyDeviceToHost));
    if(h_isITST < 0)
    {
        return true;
    }
    return false;
}

bool GIPC::checkEdgeTriIntersectionIfAny(device_TetraData& TetMesh)
{
    _scalar_scratch.resize_discard(1);
    return edgeTriIntersectionQuery(bvh_e._btype,
                                    TetMesh.vertexes,
                                    bvh_e._edges,
                                    bvh_f._faces,
                                    bvh_e._bvs,
                                    bvh_e._nodes,
                                    _scalar_scratch.data(),
                                    dHat,
                                    bvh_e.edge_number,
                                    bvh_f.face_number);
}

bool GIPC::checkGroundIntersection()
{
    int                numbers   = h_gpNum;
    if(numbers < 1)
        return false;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //

    _scalar_scratch.resize_discard(1);
    CUDA_SAFE_CALL(cudaMemset(_scalar_scratch.data(), 0, sizeof(int)));
    _checkGroundIntersection<<<blockNum, threadNum>>>(
        _vertexes, _groundOffset, _groundNormal, _environment_collisionPair, _scalar_scratch.data(), numbers);

    int h_isITST;
    CUDA_SAFE_CALL(cudaMemcpy(
        &h_isITST, _scalar_scratch.data(), sizeof(int), cudaMemcpyDeviceToHost));
    if(h_isITST < 0)
    {
        return true;
    }
    return false;
}

bool GIPC::isIntersected(device_TetraData& TetMesh)
{
    if(checkGroundIntersection())
    {
        return true;
    }

    if(checkEdgeTriIntersectionIfAny(TetMesh))
    {
        return true;
    }
    return false;
}

bool GIPC::lineSearch(device_TetraData& TetMesh, double& alpha, const double& cfl_alpha)
{
    bool stopped = false;
    //buildCP();
    double lastEnergyVal = computeEnergy(TetMesh);
    double c1m           = 0.0;
    double armijoParam   = 0;
    if(armijoParam > 0.0)
    {
        c1m += armijoParam * Energy_Add_Reduction_Algorithm(3, TetMesh);
    }

    CUDA_SAFE_CALL(cudaMemcpy(TetMesh.temp_double3Mem,
                              TetMesh.vertexes,
                              vertexNum * sizeof(double3),
                              cudaMemcpyDeviceToDevice));

    stepForward(TetMesh.vertexes, TetMesh.temp_double3Mem, _moveDir, TetMesh.BoundaryType, alpha, false, vertexNum);

    buildBVH();
    //buildCP();
    //if (h_cpNum[0] > 0) system("pause");
    int insectNum      = 0;

    bool checkInterset = true;

    while(checkInterset && isIntersected(TetMesh))
    {
        printf("type 0 intersection happened:  %d\n", insectNum);
        insectNum++;
        alpha /= 2.0;
        alpha = __m_min(cfl_alpha, alpha);
        stepForward(TetMesh.vertexes, TetMesh.temp_double3Mem, _moveDir, TetMesh.BoundaryType, alpha, false, vertexNum);
        buildBVH();
    }

    buildCP();
    //if (h_cpNum[0] > 0) system("pause");
    //rehash = false;

    //buildCollisionSets(mesh, sh, gd, true);
    double testingE = computeEnergy(TetMesh);

    int    numOfLineSearch = 0;
    double LFStepSize      = alpha;
    //double temp_c1m = c1m;
    std::cout.precision(18);
    //std::cout << "testE:    " << testingE << "      lastEnergyVal:        " << abs(lastEnergyVal- RestNHEnergy) << std::endl;
    while((testingE > lastEnergyVal + c1m * alpha) && alpha > 1e-3 * LFStepSize)
    {
        //printf("testE:    %f      lastEnergyVal:        %f         clm*alpha:    %f\n", testingE, lastEnergyVal, c1m * alpha);
        //std::cout <<numOfLineSearch<<  "   testE:    " << testingE << "      lastEnergyVal:        " << lastEnergyVal << std::endl;
        alpha /= 2.0;
        ++numOfLineSearch;

        stepForward(TetMesh.vertexes, TetMesh.temp_double3Mem, _moveDir, TetMesh.BoundaryType, alpha, false, vertexNum);

        buildBVH();
        buildCP();
        testingE = computeEnergy(TetMesh);
    }
    if(numOfLineSearch > 8)
        printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    if(alpha < LFStepSize)
    {
        bool needRecomputeCS = false;
        while(checkInterset && isIntersected(TetMesh))
        {
            printf("type 1 intersection happened:  %d\n", insectNum);
            insectNum++;
            alpha /= 2.0;
            alpha = __m_min(cfl_alpha, alpha);
            stepForward(TetMesh.vertexes,
                        TetMesh.temp_double3Mem,
                        _moveDir,
                        TetMesh.BoundaryType,
                        alpha,
                        false,
                        vertexNum);
            buildBVH();
            needRecomputeCS = true;
        }
        if(needRecomputeCS)
        {
            buildCP();
        }
    }
    //printf("    lineSearch time step:  %f\n", alpha);


    return stopped;
}


void GIPC::postLineSearch(device_TetraData& TetMesh, double alpha)
{
    if(Kappa == 0.0)
    {
        initKappa(TetMesh);
    }
    else
    {

        bool updateKappa = checkCloseGroundVal();
        if(!updateKappa)
        {
            updateKappa = checkSelfCloseVal();
        }
        if(updateKappa)
        {
            Kappa *= 2.0;
            upperBoundKappa(Kappa);
        }
        tempFree_closeConstraint();
        tempMalloc_closeConstraint();
        CUDA_SAFE_CALL(cudaMemset(_close_cpNum, 0, sizeof(uint32_t)));
        CUDA_SAFE_CALL(cudaMemset(_close_gpNum, 0, sizeof(uint32_t)));

        computeCloseGroundVal();

        computeSelfCloseVal();
    }
    //printf("------------------------------------------Kappa: %f\n", Kappa);
}

void GIPC::tempMalloc_closeConstraint()
{
    _closeConstraintID.resize_discard(h_gpNum);
    _closeConstraintVal.resize_discard(h_gpNum);
    _closeMConstraintID.resize_discard(h_cpNum[0]);
    _closeMConstraintVal.resize_discard(h_cpNum[0]);
}

void GIPC::tempFree_closeConstraint()
{
    _closeConstraintID.clear();
    _closeConstraintVal.clear();
    _closeMConstraintID.clear();
    _closeMConstraintVal.clear();
}

void GIPC::resizeFrictionBuffers()
{
    const size_t self_count   = h_cpNum[0];
    const size_t ground_count = h_gpNum;
    lambda_lastH_scalar.resize_discard(self_count);
    distCoord.resize_discard(self_count);
    tanBasis.resize_discard(self_count);
    _collisonPairs_lastH.resize_discard(self_count);
    _MatIndex_last.resize_discard(self_count);
    lambda_lastH_scalar_gd.resize_discard(ground_count);
    _collisonPairs_lastH_gd.resize_discard(ground_count);
}
double maxCOllisionPairNum = 0;
double totalCollisionPairs = 0;
double total_Cg_count      = 0;
double timemakePd          = 0;
#include <vector>
#include <fstream>
std::vector<int> iterV;
int              GIPC::solve_subIP(device_TetraData& TetMesh,
                      double&           time0,
                      double&           time1,
                      double&           time2,
                      double&           time3,
                      double&           time4)
{
    int iterCap = 10000, k = 0;
    ensureTimingEvents();
    cudaEvent_t start = _newton_timing_events[0];
    cudaEvent_t end0  = _newton_timing_events[1];
    cudaEvent_t end1  = _newton_timing_events[2];
    cudaEvent_t end2  = _newton_timing_events[3];
    cudaEvent_t end3  = _newton_timing_events[4];
    cudaEvent_t end4  = _newton_timing_events[5];

    CUDA_SAFE_CALL(cudaMemset(_moveDir, 0, vertexNum * sizeof(double3)));
    //BH.MALLOC_DEVICE_MEM_O(tetrahedraNum, h_cpNum + 1, h_gpNum);
    for(; k < iterCap; ++k)
    {
        totalCollisionPairs += h_cpNum[0];
        maxCOllisionPairNum =
            (maxCOllisionPairNum > h_cpNum[0]) ? maxCOllisionPairNum : h_cpNum[0];
        BH.updateDNum(triangleNum,
                      tetrahedraNum,
                      h_cpNum + 1,
                      h_cpNum_last + 1,
                      tri_edge_num,
                      h_gpNum,
                      h_gpNum_last,
                      softNum);

        //printf("collision num  %d\n", h_cpNum[0]);

        CUDA_SAFE_CALL(cudaEventRecord(start));
        timemakePd += computeGradientAndHessian(TetMesh);


        double distToOpt_PN = calcMinMovement(_moveDir, pcg_data.squeue, vertexNum);

        bool gradVanish = (distToOpt_PN < sqrt(Newton_solver_threshold * Newton_solver_threshold
                                               * bboxDiagSize2 * IPC_dt * IPC_dt));
        if(k && gradVanish)
        {
            break;
        }
        CUDA_SAFE_CALL(cudaEventRecord(end0));
        total_Cg_count += calculateMovingDirection(TetMesh, h_cpNum[0], pcg_data.P_type);
        CUDA_SAFE_CALL(cudaEventRecord(end1));
        double alpha = 1.0, slackness_a = 0.8, slackness_m = 0.8;

        alpha =
            __m_min(alpha, ground_largestFeasibleStepSize(slackness_a, pcg_data.squeue));
        //alpha = __m_min(alpha, InjectiveStepSize(0.2, 1e-6, pcg_data.squeue, TetMesh.tetrahedras));
        alpha             = __m_min(alpha,
                        self_largestFeasibleStepSize(slackness_m, pcg_data.squeue, h_cpNum[0]));
        double temp_alpha = alpha;
        double alpha_CFL  = alpha;

        double ccd_size = 1.0;

        buildBVH_FULLCCD(temp_alpha);
        buildFullCP(temp_alpha);
        if(h_ccd_cpNum > 0)
        {
            double maxSpeed = cfl_largestSpeed(pcg_data.squeue);
            alpha_CFL       = sqrt(dHat) / maxSpeed * 0.5;
            alpha           = __m_min(alpha, alpha_CFL);
            if(temp_alpha > 2 * alpha_CFL)
            {
                /*buildBVH_FULLCCD(temp_alpha);
                buildFullCP(temp_alpha);*/
                alpha =
                    __m_min(temp_alpha,
                            self_largestFeasibleStepSize(slackness_m, pcg_data.squeue, h_ccd_cpNum)
                                * ccd_size);
                alpha = __m_max(alpha, alpha_CFL);
            }
        }

        CUDA_SAFE_CALL(cudaEventRecord(end2));
        //printf("alpha:  %f\n", alpha);

        lineSearch(TetMesh, alpha, alpha_CFL);
        CUDA_SAFE_CALL(cudaEventRecord(end3));
        postLineSearch(TetMesh, alpha);
        CUDA_SAFE_CALL(cudaEventRecord(end4));
        //BH.FREE_DEVICE_MEM();
        //if (h_cpNum[0] > 0) return;
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        float time00, time11, time22, time33, time44;
        CUDA_SAFE_CALL(cudaEventElapsedTime(&time00, start, end0));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&time11, end0, end1));
        //total_Cg_time += time1;
        CUDA_SAFE_CALL(cudaEventElapsedTime(&time22, end1, end2));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&time33, end2, end3));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&time44, end3, end4));
        time0 += time00;
        time1 += time11;
        time2 += time22;
        time3 += time33;
        time4 += time44;
        ////*cflTime = ptime;
        //printf("time0 = %f,  time1 = %f,  time2 = %f,  time3 = %f,  time4 = %f\n", time00, time11, time22, time33, time44);
    }
    //iterV.push_back(k);
    //std::ofstream outiter("iterCount.txt");
    //for (int ii = 0;ii < iterV.size();ii++) {
    //    outiter << iterV[ii] << std::endl;
    //}
    //outiter.close();
    printf("\n\n      Kappa: %f                               iteration k:  %d\n\n\n", Kappa, k);
    return k;
}

void GIPC::updateVelocities(device_TetraData& TetMesh)
{
    int                numbers   = vertexNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _updateVelocities<<<blockNum, threadNum>>>(
        TetMesh.vertexes, TetMesh.o_vertexes, TetMesh.velocities, TetMesh.BoundaryType, IPC_dt, numbers);
}

void GIPC::updateBoundary(device_TetraData& TetMesh, double alpha)
{
    int                numbers   = vertexNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _updateBoundary<<<blockNum, threadNum>>>(
        TetMesh.vertexes, TetMesh.BoundaryType, _moveDir, alpha, numbers);
}

void GIPC::updateBoundaryMoveDir(device_TetraData& TetMesh, double alpha)
{
    int                numbers   = vertexNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _updateBoundaryMoveDir<<<blockNum, threadNum>>>(
        TetMesh.vertexes, TetMesh.BoundaryType, _moveDir, IPC_dt, FEM::PI, alpha, numbers);
}

void GIPC::updateBoundary2(device_TetraData& TetMesh)
{
    int                numbers   = vertexNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _updateBoundary2<<<blockNum, threadNum>>>(TetMesh.BoundaryType, TetMesh.Constraints, numbers);
}

void GIPC::computeXTilta(device_TetraData& TetMesh, const double& rate)
{
    int                numbers   = vertexNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _computeXTilta<<<blockNum, threadNum>>>(
        TetMesh.BoundaryType, TetMesh.velocities, TetMesh.o_vertexes, TetMesh.xTilta, IPC_dt, rate, numbers);
}

void GIPC::sortMesh(device_TetraData& TetMesh, int updateVertNum)
{
    sortGeometry(TetMesh, calcuMaxSceneSize(), updateVertNum, tetrahedraNum, triangleNum);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    updateSurfaces(TetMesh.sortMapVertIndex, _faces, updateVertNum, surface_Num);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    updateSurfaceEdges(TetMesh.sortMapVertIndex, _edges, updateVertNum, edge_Num);

    updateTriEdges_adjVerts(TetMesh.sortMapVertIndex,
                            TetMesh.tri_edges,
                            TetMesh.tri_edge_adj_vertex,
                            updateVertNum,
                            tri_edge_num);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    updateSurfaceVerts(TetMesh.sortMapVertIndex, _surfVerts, updateVertNum, surf_vertexNum);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    if(pcg_data.P_type == 1)
    {
        updateNeighborInfo(pcg_data.MP.d_neighborList,
                           pcg_data.MP.d_neighborListInit,
                           pcg_data.MP.d_neighborNum,
                           pcg_data.MP.d_neighborNumInit,
                           pcg_data.MP.d_neighborStart,
                           pcg_data.MP.d_neighborStartTemp,
                           TetMesh.sortIndex,
                           TetMesh.sortMapVertIndex,
                           updateVertNum,
                           pcg_data.MP.neighborListSize);
    }
}

int    totalNT      = 0;
double totalTime    = 0;
int    total_Frames = 0;
double ttime0       = 0;
double ttime1       = 0;
double ttime2       = 0;
double ttime3       = 0;
double ttime4       = 0;
bool   isRotate     = false;
void   GIPC::IPC_Solver(device_TetraData& TetMesh)
{
    //double animation_fullRate = 0;
    ensureTimingEvents();
    cudaEvent_t start = _frame_timing_events[0];
    cudaEvent_t end0  = _frame_timing_events[1];
    double alpha = 1;
    CUDA_SAFE_CALL(cudaEventRecord(start));
    //    if(isRotate&&total_Frames*IPC_dt>=2.2){
    //        isRotate = false;
    //        updateBoundary2(TetMesh);
    //    }
    if(isRotate)
    {
        updateBoundaryMoveDir(TetMesh, alpha);
        buildBVH_FULLCCD(alpha);
        buildFullCP(alpha);
        if(h_ccd_cpNum > 0)
        {
            double slackness_m = 0.8;
            alpha              = __m_min(alpha,
                            self_largestFeasibleStepSize(slackness_m, pcg_data.squeue, h_ccd_cpNum));
        }
        //updateBoundary(TetMesh, alpha);

        CUDA_SAFE_CALL(cudaMemcpy(TetMesh.temp_double3Mem,
                                  TetMesh.vertexes,
                                  vertexNum * sizeof(double3),
                                  cudaMemcpyDeviceToDevice));
        updateBoundaryMoveDir(TetMesh, alpha);
        stepForward(TetMesh.vertexes, TetMesh.temp_double3Mem, _moveDir, TetMesh.BoundaryType, 1, true, vertexNum);

        buildBVH();
        while(isIntersected(TetMesh))
        {
            //printf("type 0 intersection happened\n");
            alpha /= 2.0;
            updateBoundaryMoveDir(TetMesh, alpha);
            stepForward(TetMesh.vertexes,
                        TetMesh.temp_double3Mem,
                        _moveDir,
                        TetMesh.BoundaryType,
                        1,
                        true,
                        vertexNum);
            buildBVH();
        }

        buildCP();
    }


    upperBoundKappa(Kappa);
    if(Kappa < 1e-16)
    {
        suggestKappa(Kappa);
    }
    initKappa(TetMesh);

#ifdef USE_FRICTION
    resizeFrictionBuffers();
    buildFrictionSets();
#endif
    animation_fullRate = animation_subRate;
    double time0       = 0;
    double time1       = 0;
    double time2       = 0;
    double time3       = 0;
    double time4       = 0;
    while(true)
    {
        //if (h_cpNum[0] > 0) return;
        tempMalloc_closeConstraint();
        CUDA_SAFE_CALL(cudaMemset(_close_cpNum, 0, sizeof(uint32_t)));
        CUDA_SAFE_CALL(cudaMemset(_close_gpNum, 0, sizeof(uint32_t)));

        totalNT += solve_subIP(TetMesh, time0, time1, time2, time3, time4);

        double2 minMaxDist1 = minMaxGroundDist();
        double2 minMaxDist2 = minMaxSelfDist();

        double minDist = __m_min(minMaxDist1.x, minMaxDist2.x);
        double maxDist = __m_max(minMaxDist1.y, minMaxDist2.y);


        bool finishMotion = animation_fullRate > 0.99 ? true : false;
        //std::cout << "minDist:  " << minDist << "       maxDist:  " << maxDist << std::endl;
        //std::cout << "dTol:  " << dTol << "       1e-6 * bboxDiagSize2:  " << 1e-6 * bboxDiagSize2 << std::endl;
        if(finishMotion)
        {
            if((h_cpNum[0] + h_gpNum) > 0)
            {

                if(minDist < dTol)
                {
                    tempFree_closeConstraint();
                    break;
                }
                else if(maxDist < dHat)
                {
                    tempFree_closeConstraint();
                    break;
                }
                else
                {
                    tempFree_closeConstraint();
                }
            }
            else
            {
                tempFree_closeConstraint();
                break;
            }
        }
        else
        {
            tempFree_closeConstraint();
        }

        animation_fullRate += animation_subRate;
        //updateVelocities(TetMesh);

        //computeXTilta(TetMesh, 1);
#ifdef USE_FRICTION
        resizeFrictionBuffers();
        buildFrictionSets();
#endif
    }

    updateVelocities(TetMesh);

    computeXTilta(TetMesh, 1);
    CUDA_SAFE_CALL(cudaEventRecord(end0));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    float tttime;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&tttime, start, end0));
    totalTime += tttime;
    total_Frames++;
    printf("average time cost:     %f,    frame id:   %d\n", totalTime / totalNT, total_Frames);
    printf("boundary alpha: %f\n  finished a step\n", alpha);


    ttime0 += time0;
    ttime1 += time1;
    ttime2 += time2;
    ttime3 += time3;
    ttime4 += time4;


    if(std::getenv("GIPC_WRITE_TIMING"))
    {
        std::ofstream outTime("timeCost.txt");
        outTime << "time0: " << ttime0 / 1000.0 << std::endl;
        outTime << "time1: " << ttime1 / 1000.0 << std::endl;
        outTime << "time2: " << ttime2 / 1000.0 << std::endl;
        outTime << "time3: " << ttime3 / 1000.0 << std::endl;
        outTime << "time4: " << ttime4 / 1000.0 << std::endl;
        outTime << "time_makePD: " << timemakePd / 1000.0 << std::endl;
        outTime << "totalTime: " << totalTime / 1000.0 << std::endl;
        outTime << "total iter: " << totalNT << std::endl;
        outTime << "frames: " << total_Frames << std::endl;
        outTime << "totalCollisionNum: " << totalCollisionPairs << std::endl;
        outTime << "averageCollision: " << totalCollisionPairs / totalNT << std::endl;
        outTime << "maxCOllisionPairNum: " << maxCOllisionPairNum << std::endl;
        outTime << "totalCgTime: " << total_Cg_count << std::endl;
    }
}



