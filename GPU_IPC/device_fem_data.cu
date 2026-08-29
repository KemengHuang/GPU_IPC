//
// device_fem_data.cu
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include "device_fem_data.cuh"
#include "cuda_tools.h"


void device_TetraData::Malloc_DEVICE_MEM(const int& vertex_num, const int& tetradedra_num, const int& triangle_num, const int& softNum, const int& tri_edgeNum){
	int maxNumbers = vertex_num > tetradedra_num ? vertex_num : tetradedra_num;
	vertexes.resize(vertex_num);
	o_vertexes.resize(vertex_num);
	velocities.resize(vertex_num);
	rest_vertexes.resize(vertex_num);
	temp_double3Mem.resize(vertex_num);
	xTilta.resize(vertex_num);
	fb.resize(vertex_num);
	
	tetrahedras.resize(tetradedra_num);
	tempTetrahedras.resize(tetradedra_num);


	tri_edges.resize(tri_edgeNum);
	tri_edge_adj_vertex.resize(tri_edgeNum);


	volum.resize(tetradedra_num);
	masses.resize(vertex_num);

	tempDouble.resize(maxNumbers);
	//CUDA_SAFE_CALL(cudaMalloc((void**)&tempM, vertex_num * sizeof(double)));

	MChash.resize(maxNumbers);
	sortMChash.resize(maxNumbers);
	sortIndex.resize(maxNumbers);
	sortIndexOutput.resize(maxNumbers);
	BoundaryType.resize(vertex_num);
	tempBoundaryType.resize(vertex_num);

	CUDA_SAFE_CALL(cudaMemset(BoundaryType, 0, vertex_num * sizeof(int)));

	//CUDA_SAFE_CALL(cudaMalloc((void**)&sortVertIndex, vertex_num * sizeof(uint32_t)));
	sortMapVertIndex.resize(vertex_num);

	DmInverses.resize(tetradedra_num);
	Constraints.resize(vertex_num);

	tempMat3x3.resize(maxNumbers);
	//CUDA_SAFE_CALL(cudaMalloc((void**)&tempConstraints, vertex_num * sizeof(__GEIGEN__::Matrix3x3d)));

	

	targetIndex.resize(softNum);
	targetVert.resize(softNum);
	triDmInverses.resize(triangle_num);
	area.resize(triangle_num);
	triangles.resize(triangle_num);
}

device_TetraData::~device_TetraData() {
	FREE_DEVICE_MEM();
}

void device_TetraData::FREE_DEVICE_MEM() {
	sortIndex.release();
	sortMapVertIndex.release();
	vertexes.release();
	o_vertexes.release();
	temp_double3Mem.release();
	velocities.release();
	rest_vertexes.release();
	xTilta.release();
	fb.release();
	tetrahedras.release();
	tempTetrahedras.release();
	volum.release();
	masses.release();
	DmInverses.release();
	Constraints.release();
	tempMat3x3.release();
	MChash.release();
	sortMChash.release();
	sortIndexOutput.release();
	tempDouble.release();
	BoundaryType.release();
	tempBoundaryType.release();
	targetIndex.release();
	targetVert.release();
	triDmInverses.release();
	area.release();
	triangles.release();
	tri_edges.release();
	tri_edge_adj_vertex.release();
}


