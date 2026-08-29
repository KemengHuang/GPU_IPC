//
// device_fem_data.cuh
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#ifndef  __DEVICE_FEM_MESHES_CUH__
#define __DEVICE_FEM_MESHES_CUH__

//#include <cuda_runtime.h>
#include "gpu_eigen_libs.cuh"
#include "device_buffer.h"
#include <cstdint>
class device_TetraData{
public:
	gpuipc::DeviceBuffer<double3> vertexes;
	gpuipc::DeviceBuffer<double3> o_vertexes;
	gpuipc::DeviceBuffer<double3> rest_vertexes;
	gpuipc::DeviceBuffer<double3> targetVert;
	gpuipc::DeviceBuffer<double3> temp_double3Mem;
	gpuipc::DeviceBuffer<double3> velocities;
	gpuipc::DeviceBuffer<double3> xTilta;
	gpuipc::DeviceBuffer<double3> fb;
	gpuipc::DeviceBuffer<uint4> tetrahedras;
	gpuipc::DeviceBuffer<uint3> triangles;

	gpuipc::DeviceBuffer<uint2> tri_edges;
	gpuipc::DeviceBuffer<uint2> tri_edge_adj_vertex;

	gpuipc::DeviceBuffer<uint32_t> targetIndex;
	gpuipc::DeviceBuffer<uint4> tempTetrahedras;
	gpuipc::DeviceBuffer<double> volum;
	gpuipc::DeviceBuffer<double> area;
	//double* tempV;
	gpuipc::DeviceBuffer<double> masses;
	gpuipc::DeviceBuffer<double> tempDouble;
	gpuipc::DeviceBuffer<uint64_t> MChash;
	gpuipc::DeviceBuffer<uint64_t> sortMChash;
	gpuipc::DeviceBuffer<uint32_t> sortIndex;
	gpuipc::DeviceBuffer<uint32_t> sortIndexOutput;
	gpuipc::DeviceBuffer<uint32_t> sortMapVertIndex;
	//uint32_t* sortTetIndex;
	gpuipc::DeviceBuffer<__GEIGEN__::Matrix3x3d> DmInverses;
	gpuipc::DeviceBuffer<__GEIGEN__::Matrix2x2d> triDmInverses;
	gpuipc::DeviceBuffer<__GEIGEN__::Matrix3x3d> Constraints;
	gpuipc::DeviceBuffer<int> BoundaryType;
	gpuipc::DeviceBuffer<int> tempBoundaryType;
	//__GEIGEN__::Matrix3x3d* tempDmInverses;
	gpuipc::DeviceBuffer<__GEIGEN__::Matrix3x3d> tempMat3x3;

public:
	device_TetraData() {}
	~device_TetraData();
	void Malloc_DEVICE_MEM(const int& vertex_num, const int& tetradedra_num, const int& triangle_num, const int& softNum, const int& tri_edgeNum);
	void FREE_DEVICE_MEM();
};


#endif // ! __DEVICE_FEM_MESHES_CUH__
