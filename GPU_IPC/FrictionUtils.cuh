//
// FrictionUtils.cuh
// GIPC
//
// created by Kemeng Huang and Huancheng Lin on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#pragma once
#include <cuda_runtime.h>
#include "gpu_eigen_libs.cuh"
#include "math.h"
#define SFCLAMPING_ORDER 1
namespace Friction {
    using namespace __GEIGEN__;

    __device__ __host__ Matrix2x2d __M3x2_transpose_self__multiply(const Matrix3x2d& A) {
        Matrix2x2d result;

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                double temp = 0;
                for (int k = 0; k < 3; k++) {
                    temp += A.m[k][i] * A.m[k][j];
                }
                result.m[i][j] = temp;
            }
        }
        return result;
    }

    __device__ __host__ double2 __M3x2_transpose_vec3__multiply(const Matrix3x2d& A, const double3& b) {
        double x = A.m[0][0] * b.x + A.m[1][0] * b.y + A.m[2][0] * b.z;
        double y = A.m[0][1] * b.x + A.m[1][1] * b.y + A.m[2][1] * b.z;
        return make_double2(x, y);
    }

    __device__ __host__ double2 __M2x2_v2__multiply(const Matrix2x2d& A, const double2& b) {
        double x = A.m[0][0] * b.x + A.m[0][1] * b.y;
        double y = A.m[1][0] * b.x + A.m[1][1] * b.y;
        return make_double2(x, y);
    }

    __device__ __host__ void computeTangentBasis_PT(
        double3 v0,
        double3 v1,
        double3 v2,
        double3 v3,
        Matrix3x2d& basis) {
        double3 v12 = __minus(v2, v1);
        double3 v12_normalized = __normalized(v12);
        double3 c = __normalized(__v_vec_cross(__v_vec_cross(v12, __minus(v3, v1)), v12));
        basis.m[0][0] = v12_normalized.x;
        basis.m[1][0] = v12_normalized.y;
        basis.m[2][0] = v12_normalized.z;
        basis.m[0][1] = c.x;
        basis.m[1][1] = c.y;
        basis.m[2][1] = c.z;
    }
    __device__ __host__ void computeClosestPoint_PT(
        double3 v0,
        double3 v1,
        double3 v2,
        double3 v3,
        double2& beta) {
        Matrix3x2d basis;
        double3 v12 = __minus(v2, v1);
        double3 v13 = __minus(v3, v1);
        basis.m[0][0] = v12.x;
        basis.m[1][0] = v12.y;
        basis.m[2][0] = v12.z;

        basis.m[0][1] = v13.x;
        basis.m[1][1] = v13.y;
        basis.m[2][1] = v13.z;
        Matrix2x2d btb = __M3x2_transpose_self__multiply(basis);
        //Eigen::Matrix<double, 2, 3> basis;
        //basis.row(0) = v2 - v1;
        //basis.row(1) = v3 - v1;
        Matrix2x2d b2b_inv;
        __Inverse2x2(btb, b2b_inv);
        double3 v10 = __minus(v0, v1);
        double2 b = __M3x2_transpose_vec3__multiply(basis, v10);
        beta = __M2x2_v2__multiply(b2b_inv, b);
        //beta = (basis * basis.transpose()).ldlt().solve(basis * (v0 - v1).transpose());
    }
    __device__ __host__ __device__ __host__ void computeRelDX_PT(
        const double3 dx0,
        const double3 dx1,
        const double3 dx2,
        const double3 dx3,
        double beta1, double beta2,
        double3& relDX) {
        double3 b1_dx12 = __s_vec_multiply(__minus(dx2, dx1), beta1);
        double3 b2_dx13 = __s_vec_multiply(__minus(dx3, dx1), beta2);

        relDX = __minus(dx0, __add(__add(b1_dx12, b2_dx13), dx1));
    }

    __device__ __host__ void liftRelDXTanToMesh_PT(
        const double2& relDXTan,
        const Matrix3x2d& basis,
        double beta1, double beta2,
        Vector12& TTTDX) {
        double3 relDXTan3D = __M3x2_v2_multiply(basis, relDXTan);
        TTTDX.v[0] = relDXTan3D.x;
        TTTDX.v[1] = relDXTan3D.y;
        TTTDX.v[2] = relDXTan3D.z;

        TTTDX.v[3] = (-1 + beta1 + beta2) * relDXTan3D.x;
        TTTDX.v[4] = (-1 + beta1 + beta2) * relDXTan3D.y;
        TTTDX.v[5] = (-1 + beta1 + beta2) * relDXTan3D.z;

        TTTDX.v[6] = -beta1 * relDXTan3D.x;
        TTTDX.v[7] = -beta1 * relDXTan3D.y;
        TTTDX.v[8] = -beta1 * relDXTan3D.z;

        TTTDX.v[9] = -beta2 * relDXTan3D.x;
        TTTDX.v[10] = -beta2 * relDXTan3D.y;
        TTTDX.v[11] = -beta2 * relDXTan3D.z;
    }

    __device__ __host__ void computeT_PT(
        Matrix3x2d basis,
        double beta1, double beta2,
        Matrix12x2d& T) {
        T.m[0][0] = basis.m[0][0];
        T.m[1][0] = basis.m[1][0];
        T.m[2][0] = basis.m[2][0];
        T.m[0][1] = basis.m[0][1];
        T.m[1][1] = basis.m[1][1];
        T.m[2][1] = basis.m[2][1];

        T.m[3][0] = (-1 + beta1 + beta2) * basis.m[0][0];
        T.m[4][0] = (-1 + beta1 + beta2) * basis.m[1][0];
        T.m[5][0] = (-1 + beta1 + beta2) * basis.m[2][0];
        T.m[3][1] = (-1 + beta1 + beta2) * basis.m[0][1];
        T.m[4][1] = (-1 + beta1 + beta2) * basis.m[1][1];
        T.m[5][1] = (-1 + beta1 + beta2) * basis.m[2][1];

        T.m[6][0] = -beta1 * basis.m[0][0];
        T.m[7][0] = -beta1 * basis.m[1][0];
        T.m[8][0] = -beta1 * basis.m[2][0];
        T.m[6][1] = -beta1 * basis.m[0][1];
        T.m[7][1] = -beta1 * basis.m[1][1];
        T.m[8][1] = -beta1 * basis.m[2][1];

        T.m[9][0] = -beta2 * basis.m[0][0];
        T.m[10][0] = -beta2 * basis.m[1][0];
        T.m[11][0] = -beta2 * basis.m[2][0];
        T.m[9][1] = -beta2 * basis.m[0][1];
        T.m[10][1] = -beta2 * basis.m[1][1];
        T.m[11][1] = -beta2 * basis.m[2][1];

        //T.template block<3, 2>(0, 0) = basis;
        //T.template block<3, 2>(3, 0) = (-1 + beta1 + beta2) * basis;
        //T.template block<3, 2>(6, 0) = -beta1 * basis;
        //T.template block<3, 2>(9, 0) = -beta2 * basis;
    }
    __device__ __host__ void computeTangentBasis_EE(
        const double3& v0,
        const double3& v1,
        const double3& v2,
        const double3& v3,
        Matrix3x2d& basis) {
        double3 v01 = __minus(v1, v0);
        double3 v01_normalized = __normalized(v01);
        double3 c = __normalized(__v_vec_cross(__v_vec_cross(v01, __minus(v3, v2)), v01));
        basis.m[0][0] = v01_normalized.x;
        basis.m[1][0] = v01_normalized.y;
        basis.m[2][0] = v01_normalized.z;
        basis.m[0][1] = c.x;
        basis.m[1][1] = c.y;
        basis.m[2][1] = c.z;
    }
    __device__ __host__ void computeClosestPoint_EE(
        const double3& v0,
        const double3& v1,
        const double3& v2,
        const double3& v3,
        double2& gamma) {
        double3 e20 = __minus(v0, v2);
        double3 e01 = __minus(v1, v0);
        double3 e23 = __minus(v3, v2);
        Matrix2x2d coefMtr;
        coefMtr.m[0][0] = __squaredNorm(e01);
        coefMtr.m[0][1] = -__v_vec_dot(e23, e01);
        coefMtr.m[1][0] = -__v_vec_dot(e23, e01);
        coefMtr.m[1][1] = __squaredNorm(e23);

        double2 rhs;
        rhs.x = -__v_vec_dot(e20, e01);
        rhs.y = __v_vec_dot(e20, e23);
        Matrix2x2d coefMtr_inv;
        __Inverse2x2(coefMtr, coefMtr_inv);
        gamma = __M2x2_v2__multiply(coefMtr_inv, rhs);
    }
    __device__ __host__ void computeRelDX_EE(
        const double3& dx0,
        const double3& dx1,
        const double3& dx2,
        const double3& dx3,
        double gamma1, double gamma2,
        double3& relDX) {
        double3 g1_dx01 = __s_vec_multiply(__minus(dx1, dx0), gamma1);
        double3 g2_dx23 = __s_vec_multiply(__minus(dx3, dx2), gamma2);

        relDX = __minus(__add(dx0, g1_dx01), __add(dx2, g2_dx23));
    }
    __device__ __host__ void computeT_EE(
        const Matrix3x2d& basis,
        double gamma1, double gamma2,
        Matrix12x2d& T) {
        T.m[0][0] = (1.0 - gamma1) * basis.m[0][0];
        T.m[1][0] = (1.0 - gamma1) * basis.m[1][0];
        T.m[2][0] = (1.0 - gamma1) * basis.m[2][0];
        T.m[0][1] = (1.0 - gamma1) * basis.m[0][1];
        T.m[1][1] = (1.0 - gamma1) * basis.m[1][1];
        T.m[2][1] = (1.0 - gamma1) * basis.m[2][1];

        T.m[3][0] = gamma1 * basis.m[0][0];
        T.m[4][0] = gamma1 * basis.m[1][0];
        T.m[5][0] = gamma1 * basis.m[2][0];
        T.m[3][1] = gamma1 * basis.m[0][1];
        T.m[4][1] = gamma1 * basis.m[1][1];
        T.m[5][1] = gamma1 * basis.m[2][1];

        T.m[6][0] = (gamma2 - 1.0) * basis.m[0][0];
        T.m[7][0] = (gamma2 - 1.0) * basis.m[1][0];
        T.m[8][0] = (gamma2 - 1.0) * basis.m[2][0];
        T.m[6][1] = (gamma2 - 1.0) * basis.m[0][1];
        T.m[7][1] = (gamma2 - 1.0) * basis.m[1][1];
        T.m[8][1] = (gamma2 - 1.0) * basis.m[2][1];

        T.m[9][0] = -gamma2 * basis.m[0][0];
        T.m[10][0] = -gamma2 * basis.m[1][0];
        T.m[11][0] = -gamma2 * basis.m[2][0];
        T.m[9][1] = -gamma2 * basis.m[0][1];
        T.m[10][1] = -gamma2 * basis.m[1][1];
        T.m[11][1] = -gamma2 * basis.m[2][1];
    }

    __device__ __host__ void liftRelDXTanToMesh_EE(
        const double2& relDXTan,
        const Matrix3x2d& basis,
        double gamma1, double gamma2,
        Vector12& TTTDX) {
        double3 relDXTan3D = __M3x2_v2_multiply(basis, relDXTan);
        TTTDX.v[0] = (1.0 - gamma1) * relDXTan3D.x;
        TTTDX.v[1] = (1.0 - gamma1) * relDXTan3D.y;
        TTTDX.v[2] = (1.0 - gamma1) * relDXTan3D.z;

        TTTDX.v[3] = gamma1 * relDXTan3D.x;
        TTTDX.v[4] = gamma1 * relDXTan3D.y;
        TTTDX.v[5] = gamma1 * relDXTan3D.z;

        TTTDX.v[6] = (gamma2 - 1.0) * relDXTan3D.x;
        TTTDX.v[7] = (gamma2 - 1.0) * relDXTan3D.y;
        TTTDX.v[8] = (gamma2 - 1.0) * relDXTan3D.z;

        TTTDX.v[9] = -gamma2 * relDXTan3D.x;
        TTTDX.v[10] = -gamma2 * relDXTan3D.y;
        TTTDX.v[11] = -gamma2 * relDXTan3D.z;
    }

    __device__ __host__ void computeTangentBasis_PE(
        const double3& v0,
        const double3& v1,
        const double3& v2,
        Matrix3x2d& basis) {
        double3 v12 = __minus(v2, v1);
        double3 v12_normalized = __normalized(v12);
        double3 c = __normalized(__v_vec_cross(v12, __minus(v0, v1)));
        basis.m[0][0] = v12_normalized.x;
        basis.m[1][0] = v12_normalized.y;
        basis.m[2][0] = v12_normalized.z;
        basis.m[0][1] = c.x;
        basis.m[1][1] = c.y;
        basis.m[2][1] = c.z;
    }
    __device__ __host__ void computeClosestPoint_PE(
        const double3& v0,
        const double3& v1,
        const double3& v2,
        double& yita) {
        double3 e12 = __minus(v2, v1);
        yita = __v_vec_dot(__minus(v0, v1), e12) / __squaredNorm(e12);
    }
    __device__ __host__ void computeRelDX_PE(
        const double3& dx0,
        const double3& dx1,
        const double3& dx2,
        double yita,
        double3& relDX) {

        double3 y_dx12 = __s_vec_multiply(__minus(dx2, dx1), yita);

        relDX = __minus(dx0, __add(dx1, y_dx12));
    }

    __device__ __host__ void liftRelDXTanToMesh_PE(
        const double2& relDXTan,
        const Matrix3x2d& basis,
        double yita,
        Vector9& TTTDX) {
        double3 relDXTan3D = __M3x2_v2_multiply(basis, relDXTan);

        TTTDX.v[0] = relDXTan3D.x;
        TTTDX.v[1] = relDXTan3D.y;
        TTTDX.v[2] = relDXTan3D.z;

        TTTDX.v[3] = (yita - 1.0) * relDXTan3D.x;
        TTTDX.v[4] = (yita - 1.0) * relDXTan3D.y;
        TTTDX.v[5] = (yita - 1.0) * relDXTan3D.z;

        TTTDX.v[6] = -yita * relDXTan3D.x;
        TTTDX.v[7] = -yita * relDXTan3D.y;
        TTTDX.v[8] = -yita * relDXTan3D.z;
    }

    __device__ __host__ void computeT_PE(
        const Matrix3x2d& basis,
        double yita,
        Matrix9x2d& T) {

        T.m[0][0] = basis.m[0][0];
        T.m[1][0] = basis.m[1][0];
        T.m[2][0] = basis.m[2][0];
        T.m[0][1] = basis.m[0][1];
        T.m[1][1] = basis.m[1][1];
        T.m[2][1] = basis.m[2][1];

        T.m[3][0] = (yita - 1.0) * basis.m[0][0];
        T.m[4][0] = (yita - 1.0) * basis.m[1][0];
        T.m[5][0] = (yita - 1.0) * basis.m[2][0];
        T.m[3][1] = (yita - 1.0) * basis.m[0][1];
        T.m[4][1] = (yita - 1.0) * basis.m[1][1];
        T.m[5][1] = (yita - 1.0) * basis.m[2][1];

        T.m[6][0] = -yita * basis.m[0][0];
        T.m[7][0] = -yita * basis.m[1][0];
        T.m[8][0] = -yita * basis.m[2][0];
        T.m[6][1] = -yita * basis.m[0][1];
        T.m[7][1] = -yita * basis.m[1][1];
        T.m[8][1] = -yita * basis.m[2][1];
    }
    __device__ __host__ void computeTangentBasis_PP(
        const double3& v0,
        const double3& v1,
        Matrix3x2d& basis) {
        double3 v01 = __minus(v1, v0);
        double3 xCross;
        xCross.x = 0;
        xCross.y = -v01.z;
        xCross.z = v01.y;
        double3 yCross;
        yCross.x = v01.z;
        yCross.y = 0;
        yCross.z = -v01.x;

        if (__squaredNorm(xCross) > __squaredNorm(yCross)) {
            double3 xCross_n = __normalized(xCross);
            double3 c = __normalized(__v_vec_cross(v01, xCross));
            basis.m[0][0] = xCross_n.x;
            basis.m[1][0] = xCross_n.y;
            basis.m[2][0] = xCross_n.z;
            basis.m[0][1] = c.x;
            basis.m[1][1] = c.y;
            basis.m[2][1] = c.z;
        }
        else {
            double3 yCross_n = __normalized(yCross);
            double3 c = __normalized(__v_vec_cross(v01, yCross));
            basis.m[0][0] = yCross_n.x;
            basis.m[1][0] = yCross_n.y;
            basis.m[2][0] = yCross_n.z;
            basis.m[0][1] = c.x;
            basis.m[1][1] = c.y;
            basis.m[2][1] = c.z;
        }
    }
    __device__ __host__ void computeRelDX_PP(
        const double3& dx0,
        const double3& dx1,
        double3& relDX) {
        relDX = __minus(dx0, dx1);
    }

    __device__ __host__ void liftRelDXTanToMesh_PP(
        const double2& relDXTan,
        const Matrix3x2d& basis,
        Vector6& TTTDX) {
        double3 relDXTan3D = __M3x2_v2_multiply(basis, relDXTan);

        TTTDX.v[0] = relDXTan3D.x;
        TTTDX.v[1] = relDXTan3D.y;
        TTTDX.v[2] = relDXTan3D.z;

        TTTDX.v[3] = -relDXTan3D.x;
        TTTDX.v[4] = -relDXTan3D.y;
        TTTDX.v[5] = -relDXTan3D.z;
    }

    __device__ __host__ void computeT_PP(
        const Matrix3x2d& basis,
        Matrix6x2d& T) {
        T.m[0][0] = basis.m[0][0];
        T.m[1][0] = basis.m[1][0];
        T.m[2][0] = basis.m[2][0];
        T.m[0][1] = basis.m[0][1];
        T.m[1][1] = basis.m[1][1];
        T.m[2][1] = basis.m[2][1];

        T.m[3][0] = -basis.m[0][0];
        T.m[4][0] = -basis.m[1][0];
        T.m[5][0] = -basis.m[2][0];
        T.m[3][1] = -basis.m[0][1];
        T.m[4][1] = -basis.m[1][1];
        T.m[5][1] = -basis.m[2][1];
    }
    // static friction clamping model
// C0 clamping
    __device__ __host__ void f0_SF_C0(double x2, double eps_f, double& f0) {
        f0 = x2 / (2.0 * eps_f) + eps_f / 2.0;
    }

    __device__ __host__ void f1_SF_div_relDXNorm_C0(double eps_f, double& result) {
        result = 1.0 / eps_f;
    }

    __device__ __host__ void f2_SF_C0(double eps_f, double& f2) {
        f2 = 1.0 / eps_f;
    }

    // C1 clamping
    __device__ __host__ void f0_SF_C1(double x2, double eps_f, double& f0) {
        f0 = x2 * (-sqrt(x2) / 3.0 + eps_f) / (eps_f * eps_f) + eps_f / 3.0;
    }

    __device__ __host__ void f1_SF_div_relDXNorm_C1(double x2, double eps_f, double& result) {
        result = (-sqrt(x2) + 2.0 * eps_f) / (eps_f * eps_f);
    }

    __device__ __host__ void f2_SF_C1(double x2, double eps_f, double& f2) {
        f2 = 2.0 * (eps_f - sqrt(x2)) / (eps_f * eps_f);
    }

    // C2 clamping
    __device__ __host__ void f0_SF_C2(double x2, double eps_f, double& f0) {
        f0 = x2 * (0.25 * x2 - (sqrt(x2) - 1.5 * eps_f) * eps_f) / (eps_f * eps_f * eps_f) + eps_f / 4.0;
    }

    __device__ __host__ void f1_SF_div_relDXNorm_C2(double x2, double eps_f, double& result) {
        result = (x2 - (3.0 * sqrt(x2) - 3.0 * eps_f) * eps_f) / (eps_f * eps_f * eps_f);
    }

    __device__ __host__ void f2_SF_C2(double x2, double eps_f, double& f2) {
        f2 = 3.0 * (x2 - (2.0 * sqrt(x2) - eps_f) * eps_f) / (eps_f * eps_f * eps_f);
    }

    // interfaces
    __device__ __host__ void f0_SF(double relDXSqNorm, double eps_f, double& f0) {
#if (SFCLAMPING_ORDER == 0)
        f0_SF_C0(relDXSqNorm, eps_f, f0);
#elif (SFCLAMPING_ORDER == 1)
        f0_SF_C1(relDXSqNorm, eps_f, f0);
#elif (SFCLAMPING_ORDER == 2)
        f0_SF_C2(relDXSqNorm, eps_f, f0);
#endif
    }

    __device__ __host__ void f1_SF_div_relDXNorm(double relDXSqNorm, double eps_f, double& result) {
#if (SFCLAMPING_ORDER == 0)
        f1_SF_div_relDXNorm_C0(eps_f, result);
#elif (SFCLAMPING_ORDER == 1)
        f1_SF_div_relDXNorm_C1(relDXSqNorm, eps_f, result);
#elif (SFCLAMPING_ORDER == 2)
        f1_SF_div_relDXNorm_C2(relDXSqNorm, eps_f, result);
#endif
    }

    __device__ __host__ void f2_SF(double relDXSqNorm, double eps_f, double& f2) {
#if (SFCLAMPING_ORDER == 0)
        f2_SF_C0(eps_f, f2);
#elif (SFCLAMPING_ORDER == 1)
        f2_SF_C1(relDXSqNorm, eps_f, f2);
#elif (SFCLAMPING_ORDER == 2)
        f2_SF_C2(relDXSqNorm, eps_f, f2);
#endif
    }
}