//
// GIPC.cuh
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#pragma once
#ifndef _GIPC_H_
#define _GIPC_H_
#include "mlbvh.cuh"
#include "device_fem_data.cuh"
#include "device_buffer.h"

#include "PCG_SOLVER.cuh"


class GIPC {
public:
	bool animation = false;
	double3* _vertexes = nullptr;
	double3* _rest_vertexes = nullptr;
	gpuipc::DeviceBuffer<uint3> _faces;
	gpuipc::DeviceBuffer<uint2> _edges;
	gpuipc::DeviceBuffer<uint32_t> _surfVerts;


	double3* targetVert = nullptr;
	uint32_t* targetInd = nullptr;
	uint32_t softNum = 0;
	uint32_t triangleNum = 0;


	double3* _moveDir = nullptr;
	lbvh_f bvh_f;
	lbvh_e bvh_e;

	PCG_Data pcg_data;

	gpuipc::DeviceBuffer<int4> _collisonPairs;
	gpuipc::DeviceBuffer<int4> _ccd_collisonPairs;
	gpuipc::DeviceBuffer<uint32_t> _cpNum;
	gpuipc::DeviceBuffer<int> _MatIndex;
	gpuipc::DeviceBuffer<uint32_t> _close_cpNum;

	gpuipc::DeviceBuffer<uint32_t> _environment_collisionPair;

	gpuipc::DeviceBuffer<uint32_t> _closeConstraintID;
	gpuipc::DeviceBuffer<double> _closeConstraintVal;

	gpuipc::DeviceBuffer<int4> _closeMConstraintID;
	gpuipc::DeviceBuffer<double> _closeMConstraintVal;

	gpuipc::DeviceBuffer<uint32_t> _gpNum;
	gpuipc::DeviceBuffer<uint32_t> _close_gpNum;
	gpuipc::DeviceBuffer<int> _scalar_scratch;
	gpuipc::DeviceBuffer<double2> _distance_scratch;
	cudaEvent_t _frame_timing_events[2] = {};
	cudaEvent_t _newton_timing_events[6] = {};
	//uint32_t* _cpNum;
	uint32_t h_cpNum[5] = {0, 0, 0, 0, 0};
	uint32_t h_ccd_cpNum = 0;
	uint32_t h_gpNum = 0;
	
	uint32_t h_close_cpNum = 0;
	uint32_t h_close_gpNum = 0;

	double Kappa;
	double dHat;
	double fDhat;
	double bboxDiagSize2;
	double relative_dhat;
	double dTol;
    double minKappaCoef;
	double IPC_dt;
	double Step;
	double meanMass;
	double meanVolumn;
	gpuipc::DeviceBuffer<double3> _groundNormal;
	gpuipc::DeviceBuffer<double> _groundOffset;

	// for friction
	gpuipc::DeviceBuffer<double> lambda_lastH_scalar;
	gpuipc::DeviceBuffer<double2> distCoord;
	gpuipc::DeviceBuffer<__GEIGEN__::Matrix3x2d> tanBasis;
	gpuipc::DeviceBuffer<int4> _collisonPairs_lastH;
	uint32_t h_cpNum_last[5];
	gpuipc::DeviceBuffer<int> _MatIndex_last;

	gpuipc::DeviceBuffer<double> lambda_lastH_scalar_gd;
	gpuipc::DeviceBuffer<uint32_t> _collisonPairs_lastH_gd;
	uint32_t h_gpNum_last = 0;

	uint32_t vertexNum = 0;
	uint32_t surf_vertexNum = 0;
	uint32_t edge_Num = 0;
	uint32_t tri_edge_num = 0;
	uint32_t surface_Num = 0;
	uint32_t tetrahedraNum = 0;

	BHessian BH;
	AABB SceneSize;
	int MAX_COLLITION_PAIRS_NUM = 0;
	int MAX_CCD_COLLITION_PAIRS_NUM = 0;

	//double RestNHEnergy;
	double animation_subRate;
	double animation_fullRate;


	double bendStiff;



	double density;
	double YoungModulus;
	double PoissonRate;
	double lengthRateLame;
	double volumeRateLame;
	double lengthRate;
	double volumeRate;
	double frictionRate;
	double clothThickness;
	double clothYoungModulus;
	double stretchStiff;
	double shearStiff;
	double clothDensity;
	double softMotionRate;
	double Newton_solver_threshold;
	double pcg_threshold;
	//bool USE_MAS;
public:
	GIPC();
	~GIPC();
	void MALLOC_DEVICE_MEM();

	void tempMalloc_closeConstraint();
	void tempFree_closeConstraint();
	void resizeFrictionBuffers();
	void ensureTimingEvents();
	void releaseTimingEvents();

	void FREE_DEVICE_MEM();
	void initBVH(int* _btype);
	void init(double m_meanMass, double m_meanVolumn);

	void buildCP();
	void buildFullCP(const double& alpha);
	void buildBVH();

	AABB* calcuMaxSceneSize();

	void buildBVH_FULLCCD(const double& alpha);


	void GroundCollisionDetect();
    void calBarrierGradientAndHessian(double3* _gradient, double mKappa);
	void calBarrierGradient(double3* _gradient, double mKap);
	void calFrictionHessian(device_TetraData& TetMesh);
	void calFrictionGradient(double3* _gradient, device_TetraData& TetMesh);

	int calculateMovingDirection(device_TetraData& TetMesh, int cpNum, int preconditioner_type = 0);
	float computeGradientAndHessian(device_TetraData& TetMesh);
	void computeGroundGradientAndHessian(double3* _gradient);
	void computeGroundGradient(double3* _gradient, double mKap);
	void computeSoftConstraintGradientAndHessian(double3* _gradient);
	void computeSoftConstraintGradient(double3* _gradient);
	double computeEnergy(device_TetraData& TetMesh);

	double Energy_Add_Reduction_Algorithm(int type, device_TetraData& TetMesh);

	double ground_largestFeasibleStepSize(double slackness, double* mqueue);

	double self_largestFeasibleStepSize(double slackness, double* mqueue, int numbers);

	double InjectiveStepSize(double slackness, double errorRate, double* mqueue, uint4* tets);

	double cfl_largestSpeed(double* mqueue);

	bool lineSearch(device_TetraData& TetMesh, double& alpha, const double& cfl_alpha);
	void postLineSearch(device_TetraData& TetMesh, double alpha);

	bool checkEdgeTriIntersectionIfAny(device_TetraData& TetMesh);
	bool isIntersected(device_TetraData& TetMesh);
	bool checkGroundIntersection();

	void computeCloseGroundVal();
	void computeSelfCloseVal();

	bool checkCloseGroundVal();
	bool checkSelfCloseVal();

	double2 minMaxGroundDist();
	double2 minMaxSelfDist();

	void updateVelocities(device_TetraData& TetMesh);
	void updateBoundary(device_TetraData& TetMesh, double alpha);
	void updateBoundaryMoveDir(device_TetraData& TetMesh, double alpha);
    void updateBoundary2(device_TetraData& TetMesh);
	void computeXTilta(device_TetraData& TetMesh, const double& rate);

	void initKappa(device_TetraData& TetMesh);
	void suggestKappa(double& kappa);
	void upperBoundKappa(double& kappa);
	int solve_subIP(device_TetraData& TetMesh, double& time0, double& time1, double& time2, double& time3, double& time4);
	void IPC_Solver(device_TetraData& TetMesh);
	void sortMesh(device_TetraData& TetMesh, int updateVertNum);
	void buildFrictionSets();

};

#endif
