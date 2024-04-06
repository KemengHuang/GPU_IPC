//
// load_mesh.h
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#pragma once
#ifndef FEM_MESH_H
#define FEM_MESH_H
//#include "Eigen/Eigen"
//#include "mIPC.h"
#include <vector>
#include <cuda_runtime.h>
#include <string>
#include <sstream>
#include "eigen_data.h"
using namespace std;


class mesh_obj {
public:
	vector<double3> vertexes;
	vector<double3> normals;
	vector<uint3> facenormals;
	vector<uint3> faces;
	vector<uint2> edges;
	int vertexNum;
	int faceNum;
	int edgeNum;
	//void InitMesh(int type, double scale);
	bool load_mesh(const std::string& filename, double scale, double3 transform);
};

class tetrahedra_obj {
public:
	double maxVolum;
	vector<bool> isNBC;
	vector<bool> isCollide;
	vector<double> volum;
	vector<double> area;
	vector<double> masses;

	double meanMass;
	double meanVolum;
	vector<double3> vertexes;
	vector<int> boundaryTypies;
	vector<uint4> tetrahedras;
	vector<uint3> triangles;
	vector<uint32_t> targetIndex;
	vector<double3> forces;
	vector<double3> velocities;
	vector<double3> d_velocities;
	vector<__GEIGEN__::Matrix3x3d> DM_inverse;
	vector<__GEIGEN__::Matrix2x2d> tri_DM_inverse;
	vector<__GEIGEN__::Matrix3x3d> constraints;

	vector<double3> targetPos;
	vector<double3> tetra_fiberDir;

	std::vector<uint2> tri_edges_adj_points;
	std::vector<uint2> tri_edges;


	vector<uint32_t> surfId2TetId;
	vector<uint3> surface;

	vector<uint32_t> surfVerts;
	vector<uint2> surfEdges;

	vector<double3> xTilta, dx_Elastic, acceleration;
	vector<double3> rest_V, V_prev;

	vector<vector<unsigned int>> vertNeighbors;
	vector<unsigned int> neighborList;
	vector<unsigned int> neighborStart;
	vector<unsigned int> neighborNum;

	int D12x12Num;
	int D9x9Num;
	int D6x6Num;
	int D3x3Num;
	int vertexNum;
	int tetrahedraNum;
	int triangleNum;
	int softNum;
	int vertexOffset;

	double3 minTConer;
	double3 maxTConer;

	double3 minConer;
	double3 maxConer;
	tetrahedra_obj();
	int getVertNeighbors();
	//void InitMesh(int type, double scale);
	bool load_tetrahedraMesh(const std::string& filename, double scale, double3 position_offset);
	bool load_triMesh(const std::string& filename, double scale, double3 transform, int boundaryType);
	bool load_animation(const std::string& filename, double scale, double3 transform);
	bool load_tetrahedraMesh_IPC_TetMesh(const std::string& filename, double scale, double3 position_offset);
	//void load_test(double scale, int num = 1);
	void getSurface();
	bool output_tetrahedraMesh(const std::string& filename);
};

#endif // !FEM_MESH.H