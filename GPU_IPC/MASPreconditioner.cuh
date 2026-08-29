//
// MASPreconditioner.cuh
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include "device_fem_data.cuh"
#include "eigen_data.h"
#include "device_buffer.h"

class BHessian {
public:
	gpuipc::DeviceBuffer<uint32_t> D1Index;
	gpuipc::DeviceBuffer<uint3> D3Index;
	gpuipc::DeviceBuffer<uint4> D4Index;
	gpuipc::DeviceBuffer<uint2> D2Index;
	gpuipc::DeviceBuffer<__GEIGEN__::Matrix12x12d> H12x12;
	gpuipc::DeviceBuffer<__GEIGEN__::Matrix3x3d> H3x3;
	gpuipc::DeviceBuffer<__GEIGEN__::Matrix6x6d> H6x6;
	gpuipc::DeviceBuffer<__GEIGEN__::Matrix9x9d> H9x9;

	uint32_t DNum[4] = {0, 0, 0, 0};

public:
	BHessian() {}
	~BHessian() {};
	void updateDNum(const int& tri_Num,
	                const int& tet_number,
	                const uint32_t* cpNums,
	                const uint32_t* last_cpNums,
	                const int& tri_edge_number,
	                uint32_t ground_count,
	                uint32_t previous_ground_count,
	                uint32_t soft_count);
	void MALLOC_DEVICE_MEM_O(const int& tet_number, const int& surfvert_number, const int& surface_number, const int& edge_number, const int& triangle_num, const int& tri_Edge_number);
	void FREE_DEVICE_MEM();
	void ensure_current_capacity();
	//void init(const int& edgeNum, const int& faceNum, const int& vertNum);
};

class MASPreconditioner {
	
	int totalNodes = 0;
	int levelnum = 0;
	//int totalSize;
	int totalNumberClusters = 0;
	//int bankSize;
	int2 h_clevelSize = make_int2(0, 0);
	int4* _collisonPairs = nullptr;

	gpuipc::DeviceBuffer<int2> d_levelSize;
	gpuipc::DeviceBuffer<int> d_coarseSpaceTables;
	gpuipc::DeviceBuffer<int> d_prefixOriginal;
	gpuipc::DeviceBuffer<int> d_prefixSumOriginal;
	gpuipc::DeviceBuffer<int> d_goingNext;
	gpuipc::DeviceBuffer<int> d_denseLevel;
	gpuipc::DeviceBuffer<int> d_coarseTable;
	gpuipc::DeviceBuffer<unsigned int> d_fineConnectMask;
	gpuipc::DeviceBuffer<unsigned int> d_nextConnectMask;
	gpuipc::DeviceBuffer<unsigned int> d_nextPrefix;
	gpuipc::DeviceBuffer<unsigned int> d_nextPrefixSum;


	gpuipc::DeviceBuffer<__GEIGEN__::Matrix96x96T> d_Mat96;
    gpuipc::DeviceBuffer<__GEIGEN__::MasMatrixSymf> d_inverseMat96;
	gpuipc::DeviceBuffer<Precision_T3> d_multiLevelR;
	gpuipc::DeviceBuffer<Precision_T3> d_multiLevelZ;

public:
	int neighborListSize = 0;
	gpuipc::DeviceBuffer<unsigned int> d_neighborList;
	gpuipc::DeviceBuffer<unsigned int> d_neighborStart;
	gpuipc::DeviceBuffer<unsigned int> d_neighborStartTemp;
	gpuipc::DeviceBuffer<unsigned int> d_neighborNum;
	gpuipc::DeviceBuffer<unsigned int> d_neighborListInit;
	gpuipc::DeviceBuffer<unsigned int> d_neighborNumInit;

public:
	void computeNumLevels(int vertNum);
	void initPreconditioner(int vertNum, int totalNeighborNum, int4* m_collisonPairs);
	void updateCollisionPairs(int4* collisionPairs) { _collisonPairs = collisionPairs; }

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

private:
	void ensureRuntimeCapacity();
};
