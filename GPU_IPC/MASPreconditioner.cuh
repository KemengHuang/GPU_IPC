//
// MASPreconditioner.cuh
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include "device_fem_data.cuh"
#include "eigen_data.h"

class BHessian {
public:
	uint32_t* D1Index;//pIndex, DpeIndex, DptIndex;
	uint3* D3Index;
	uint4* D4Index;
	uint2* D2Index;
	__GEIGEN__::Matrix12x12d* H12x12;
	__GEIGEN__::Matrix3x3d* H3x3;
	__GEIGEN__::Matrix6x6d* H6x6;
	__GEIGEN__::Matrix9x9d* H9x9;

	uint32_t DNum[4];

public:
	BHessian() {}
	~BHessian() {};
	void updateDNum(const int& tri_Num, const int& tet_number, const uint32_t* cpNums, const uint32_t* last_cpNums, const int& tri_edge_number);
	void MALLOC_DEVICE_MEM_O(const int& tet_number, const int& surfvert_number, const int& surface_number, const int& edge_number, const int& triangle_num, const int& tri_Edge_number);
	void FREE_DEVICE_MEM();
	//void init(const int& edgeNum, const int& faceNum, const int& vertNum);
};

class MASPreconditioner {
	
	int totalNodes;
	int levelnum;
	//int totalSize;
	int totalNumberClusters;
	//int bankSize;
	int2 h_clevelSize;
	int4* _collisonPairs;

	int2* d_levelSize;
	int* d_coarseSpaceTables;
	int* d_prefixOriginal;
	int* d_prefixSumOriginal;
	int* d_goingNext;
	int* d_denseLevel;
	int4* d_coarseTable;
	unsigned int* d_fineConnectMask;
	unsigned int* d_nextConnectMask;
	unsigned int* d_nextPrefix;
	unsigned int* d_nextPrefixSum;


	__GEIGEN__::Matrix96x96T* d_Mat96;
    __GEIGEN__::MasMatrixSymf* d_inverseMat96;
	Precision_T3* d_multiLevelR;
	Precision_T3* d_multiLevelZ;

public:
	int neighborListSize;
	unsigned int* d_neighborList;
	unsigned int* d_neighborStart;
	unsigned int* d_neighborStartTemp;
	unsigned int* d_neighborNum;
	unsigned int* d_neighborListInit;
	unsigned int* d_neighborNumInit;

public:
	void computeNumLevels(int vertNum);
	void initPreconditioner(int vertNum, int totalNeighborNum, int4* m_collisonPairs);

	void BuildConnectMaskL0();
	void PreparePrefixSumL0();

	void BuildLevel1();
	void BuildConnectMaskLx(int level);
	void NextLevelCluster(int level);
	void PrefixSumLx(int level);
	void ComputeNextLevel(int level);
	void AggregationKernel();

	int ReorderRealtime(int cpNum);
	void PrepareHessian(const BHessian& BH, const double* masses);

	void setPreconditioner(const BHessian& BH, const double* masses, int cpNum);

	void BuildCollisionConnection(unsigned int* connectionMsk, int* coarseTableSpace, int level, int cpNum);

	void BuildMultiLevelR(const double3* R);
	void SchwarzLocalXSym();

	void CollectFinalZ(double3* Z);

	void preconditioning(const double3* R, double3* Z);

	void FreeMAS();
};