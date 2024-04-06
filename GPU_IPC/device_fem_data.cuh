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
#include <cstdint>
class device_TetraData{
public:
	double3* vertexes;
	double3* o_vertexes;
	double3* rest_vertexes;
	double3* targetVert;
	double3* temp_double3Mem;
	double3* velocities;
	double3* xTilta;
	double3* fb;
	uint4* tetrahedras;
	uint3* triangles;

	uint2* tri_edges;
	uint2* tri_edge_adj_vertex;

	uint32_t* targetIndex;
	uint4* tempTetrahedras;
	double* volum;
	double* area;
	//double* tempV;
	double* masses;
	double* tempDouble;
	uint64_t* MChash;
	uint32_t* sortIndex;
	uint32_t* sortMapVertIndex;
	//uint32_t* sortTetIndex;
	__GEIGEN__::Matrix3x3d* DmInverses;
	__GEIGEN__::Matrix2x2d* triDmInverses;
	__GEIGEN__::Matrix3x3d* Constraints;
	int* BoundaryType;
	int* tempBoundaryType;
	//__GEIGEN__::Matrix3x3d* tempDmInverses;
	__GEIGEN__::Matrix3x3d* tempMat3x3;

public:
	device_TetraData() {}
	~device_TetraData();
	void Malloc_DEVICE_MEM(const int& vertex_num, const int& tetradedra_num, const int& triangle_num, const int& softNum, const int& tri_edgeNum);
	void FREE_DEVICE_MEM();
};


#endif // ! __DEVICE_FEM_MESHES_CUH__
