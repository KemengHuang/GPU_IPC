//
// femEnergy.cu
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include "femEnergy.cuh"
#include "math.h"
#include <stdio.h>
#include "Eigen/Eigen"
#include "device_utils.h"
namespace LibShell {
    __device__ __host__
        static Eigen::Matrix3d crossMatrix(Eigen::Vector3d v) {
        Eigen::Matrix3d ret;
        ret << 0, -v[2], v[1],
            v[2], 0, -v[0],
            -v[1], v[0], 0;
        return ret;
    }

    __device__ __host__
        static Eigen::Matrix2d adjugate(Eigen::Matrix2d M) {
        Eigen::Matrix2d ret;
        ret << M(1, 1), -M(0, 1), -M(1, 0), M(0, 0);
        return ret;
    }
    __device__ __host__
        static double angle(const Eigen::Vector3d& v, const Eigen::Vector3d& w, const Eigen::Vector3d& axis,
            Eigen::Matrix<double, 1, 9>* derivative, // v, w
            Eigen::Matrix<double, 9, 9>* hessian
        ) {
        double theta = 2.0 * atan2((v.cross(w).dot(axis) / axis.norm()), v.dot(w) + v.norm() * w.norm());

        if (derivative) {
            derivative->segment<3>(0) = -axis.cross(v) / v.squaredNorm() / axis.norm();
            derivative->segment<3>(3) = axis.cross(w) / w.squaredNorm() / axis.norm();
            derivative->segment<3>(6).setZero();
        }
        if (hessian) {
            hessian->setZero();
            hessian->block<3, 3>(0, 0) +=
                2.0 * (axis.cross(v)) * v.transpose() / v.squaredNorm() / v.squaredNorm() / axis.norm();
            hessian->block<3, 3>(3, 3) +=
                -2.0 * (axis.cross(w)) * w.transpose() / w.squaredNorm() / w.squaredNorm() / axis.norm();
            hessian->block<3, 3>(0, 0) += -crossMatrix(axis) / v.squaredNorm() / axis.norm();
            hessian->block<3, 3>(3, 3) += crossMatrix(axis) / w.squaredNorm() / axis.norm();

            Eigen::Matrix3d dahat = (Eigen::Matrix3d::Identity() / axis.norm() -
                axis * axis.transpose() / axis.norm() / axis.norm() / axis.norm());

            hessian->block<3, 3>(0, 6) += crossMatrix(v) * dahat / v.squaredNorm();
            hessian->block<3, 3>(3, 6) += -crossMatrix(w) * dahat / w.squaredNorm();
        }

        return theta;
    }
    __device__ __host__
        static double edgeTheta(
            const Eigen::Vector3d& q0,
            const Eigen::Vector3d& q1,
            const Eigen::Vector3d& q2,
            const Eigen::Vector3d& q3,
            Eigen::Matrix<double, 1, 12>* derivative, // edgeVertex, then edgeOppositeVertex
            Eigen::Matrix<double, 12, 12>* hessian) {
        if (derivative)
            derivative->setZero();
        if (hessian)
            hessian->setZero();
        //    int v0 = mesh.edgeVertex(edge, 0);
        //    int v1 = mesh.edgeVertex(edge, 1);
        //    int v2 = mesh.edgeOppositeVertex(edge, 0);
        //    int v3 = mesh.edgeOppositeVertex(edge, 1);
        //    if (v2 == -1 || v3 == -1)
        //        return 0; // boundary edge
        //
        //    Eigen::Vector3d q0 = curPos.row(v0);
        //    Eigen::Vector3d q1 = curPos.row(v1);
        //    Eigen::Vector3d q2 = curPos.row(v2);
        //    Eigen::Vector3d q3 = curPos.row(v3);

        Eigen::Vector3d n0 = (q0 - q2).cross(q1 - q2);
        Eigen::Vector3d n1 = (q1 - q3).cross(q0 - q3);
        Eigen::Vector3d axis = q1 - q0;
        Eigen::Matrix<double, 1, 9> angderiv;
        Eigen::Matrix<double, 9, 9> anghess;

        double theta = angle(n0, n1, axis, (derivative || hessian) ? &angderiv : NULL, hessian ? &anghess : NULL);

        if (derivative) {
            derivative->block<1, 3>(0, 0) += angderiv.block<1, 3>(0, 0) * crossMatrix(q2 - q1);
            derivative->block<1, 3>(0, 3) += angderiv.block<1, 3>(0, 0) * crossMatrix(q0 - q2);
            derivative->block<1, 3>(0, 6) += angderiv.block<1, 3>(0, 0) * crossMatrix(q1 - q0);

            derivative->block<1, 3>(0, 0) += angderiv.block<1, 3>(0, 3) * crossMatrix(q1 - q3);
            derivative->block<1, 3>(0, 3) += angderiv.block<1, 3>(0, 3) * crossMatrix(q3 - q0);
            derivative->block<1, 3>(0, 9) += angderiv.block<1, 3>(0, 3) * crossMatrix(q0 - q1);
        }

        if (hessian) {
            Eigen::Matrix3d vqm[3];
            vqm[0] = crossMatrix(q0 - q2);
            vqm[1] = crossMatrix(q1 - q0);
            vqm[2] = crossMatrix(q2 - q1);
            Eigen::Matrix3d wqm[3];
            wqm[0] = crossMatrix(q0 - q1);
            wqm[1] = crossMatrix(q1 - q3);
            wqm[2] = crossMatrix(q3 - q0);

            int vindices[3] = { 3, 6, 0 };
            int windices[3] = { 9, 0, 3 };

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    hessian->block<3, 3>(vindices[i], vindices[j]) +=
                        vqm[i].transpose() * anghess.block<3, 3>(0, 0) * vqm[j];
                    hessian->block<3, 3>(vindices[i], windices[j]) +=
                        vqm[i].transpose() * anghess.block<3, 3>(0, 3) * wqm[j];
                    hessian->block<3, 3>(windices[i], vindices[j]) +=
                        wqm[i].transpose() * anghess.block<3, 3>(3, 0) * vqm[j];
                    hessian->block<3, 3>(windices[i], windices[j]) +=
                        wqm[i].transpose() * anghess.block<3, 3>(3, 3) * wqm[j];
                }

                hessian->block<3, 3>(vindices[i], 3) += vqm[i].transpose() * anghess.block<3, 3>(0, 6);
                hessian->block<3, 3>(3, vindices[i]) += anghess.block<3, 3>(6, 0) * vqm[i];
                hessian->block<3, 3>(vindices[i], 0) += -vqm[i].transpose() * anghess.block<3, 3>(0, 6);
                hessian->block<3, 3>(0, vindices[i]) += -anghess.block<3, 3>(6, 0) * vqm[i];

                hessian->block<3, 3>(windices[i], 3) += wqm[i].transpose() * anghess.block<3, 3>(3, 6);
                hessian->block<3, 3>(3, windices[i]) += anghess.block<3, 3>(6, 3) * wqm[i];
                hessian->block<3, 3>(windices[i], 0) += -wqm[i].transpose() * anghess.block<3, 3>(3, 6);
                hessian->block<3, 3>(0, windices[i]) += -anghess.block<3, 3>(6, 3) * wqm[i];

            }

            Eigen::Vector3d dang1 = angderiv.block<1, 3>(0, 0).transpose();
            Eigen::Vector3d dang2 = angderiv.block<1, 3>(0, 3).transpose();

            Eigen::Matrix3d dang1mat = crossMatrix(dang1);
            Eigen::Matrix3d dang2mat = crossMatrix(dang2);

            hessian->block<3, 3>(6, 3) += dang1mat;
            hessian->block<3, 3>(0, 3) -= dang1mat;
            hessian->block<3, 3>(0, 6) += dang1mat;
            hessian->block<3, 3>(3, 0) += dang1mat;
            hessian->block<3, 3>(3, 6) -= dang1mat;
            hessian->block<3, 3>(6, 0) -= dang1mat;

            hessian->block<3, 3>(9, 0) += dang2mat;
            hessian->block<3, 3>(3, 0) -= dang2mat;
            hessian->block<3, 3>(3, 9) += dang2mat;
            hessian->block<3, 3>(0, 3) += dang2mat;
            hessian->block<3, 3>(0, 9) -= dang2mat;
            hessian->block<3, 3>(9, 3) -= dang2mat;
        }

        return theta;
    }

};



__device__ __host__
void __calculateDm2D_double(const double3* vertexes, const uint3& index, __GEIGEN__::Matrix2x2d& M) {
    double3 v01 = __GEIGEN__::__minus(vertexes[index.y], vertexes[index.x]);
    double3 v02 = __GEIGEN__::__minus(vertexes[index.z], vertexes[index.x]);
    double3 normal = __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(v01, v02));
    double3 target; target.x = 0; target.y = 0; target.z = 1;
    double3 vec = __GEIGEN__::__v_vec_cross(normal, target);
    double cos = __GEIGEN__::__v_vec_dot(normal, target);
    __GEIGEN__::Matrix3x3d rotation;
    __GEIGEN__::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);

    if (cos + 1 == 0) {
        rotation.m[0][0] = -1; rotation.m[1][1] = -1;
    }
    else {
        __GEIGEN__::Matrix3x3d cross_vec;
        __GEIGEN__::__set_Mat_val(cross_vec, 0, -vec.z, vec.y, vec.z, 0, -vec.x, -vec.y, vec.x, 0);
        rotation = __GEIGEN__::__Mat_add(rotation, __GEIGEN__::__Mat_add(cross_vec, __GEIGEN__::__S_Mat_multiply(__GEIGEN__::__M_Mat_multiply(cross_vec, cross_vec), 1.0 / (1 + cos))));
    }

    double3 rotate_uv0 = __GEIGEN__::__M_v_multiply(rotation, vertexes[index.x]);
    double3 rotate_uv1 = __GEIGEN__::__M_v_multiply(rotation, vertexes[index.y]);
    double3 rotate_uv2 = __GEIGEN__::__M_v_multiply(rotation, vertexes[index.z]);

    double2 uv0 = make_double2(rotate_uv0.x, rotate_uv0.y);
    double2 uv1 = make_double2(rotate_uv1.x, rotate_uv1.y);
    double2 uv2 = make_double2(rotate_uv2.x, rotate_uv2.y);

    double2 u0 = __GEIGEN__::__minus_v2(uv1, uv0);
    double2 u1 = __GEIGEN__::__minus_v2(uv2, uv0);

    __GEIGEN__::__set_Mat2x2_val_column(M, u0, u1);
}

__device__ __host__
void __calculateDs2D_double(const double3* vertexes, const uint3& index, __GEIGEN__::Matrix3x2d& M) {
    double o1x = vertexes[index.y].x - vertexes[index.x].x;
    double o1y = vertexes[index.y].y - vertexes[index.x].y;
    double o1z = vertexes[index.y].z - vertexes[index.x].z;

    double o2x = vertexes[index.z].x - vertexes[index.x].x;
    double o2y = vertexes[index.z].y - vertexes[index.x].y;
    double o2z = vertexes[index.z].z - vertexes[index.x].z;


    M.m[0][0] = o1x; M.m[0][1] = o2x;
    M.m[1][0] = o1y; M.m[1][1] = o2y;
    M.m[2][0] = o1z; M.m[2][1] = o2z;
}
__device__ __host__
void __calculateDms3D_double(const double3* vertexes, const uint4& index, __GEIGEN__::Matrix3x3d& M) {

    double o1x = vertexes[index.y].x - vertexes[index.x].x;
    double o1y = vertexes[index.y].y - vertexes[index.x].y;
    double o1z = vertexes[index.y].z - vertexes[index.x].z;

    double o2x = vertexes[index.z].x - vertexes[index.x].x;
    double o2y = vertexes[index.z].y - vertexes[index.x].y;
    double o2z = vertexes[index.z].z - vertexes[index.x].z;

    double o3x = vertexes[index.w].x - vertexes[index.x].x;
    double o3y = vertexes[index.w].y - vertexes[index.x].y;
    double o3z = vertexes[index.w].z - vertexes[index.x].z;

    M.m[0][0] = o1x; M.m[0][1] = o2x; M.m[0][2] = o3x;
    M.m[1][0] = o1y; M.m[1][1] = o2y; M.m[1][2] = o3y;
    M.m[2][0] = o1z; M.m[2][1] = o2z; M.m[2][2] = o3z;
}
__device__
__GEIGEN__::Matrix3x2d __computePEPF_BaraffWitkinStretch_double(const __GEIGEN__::Matrix3x2d& F, double stretchStiff, double shearStiff) {

    double2 u, v;
    u.x = 1; u.y = 0;
    v.x = 0; v.y = 1;
    double I5u = __GEIGEN__::__squaredNorm(__M3x2_v2_multiply(F, u));
    double I5v = __GEIGEN__::__squaredNorm(__M3x2_v2_multiply(F, v));
    double ucoeff = 1.0 - 1 / sqrt(I5u);
    double vcoeff = 1.0 - 1 / sqrt(I5v);

    if (I5u < 1) {
        ucoeff *= 1e-2;
    }
    if (I5v < 1) {
        vcoeff *= 1e-2;
    }


    __GEIGEN__::Matrix2x2d uu = __GEIGEN__::__v2_vec2_toMat2x2(u, u);
    __GEIGEN__::Matrix2x2d vv = __GEIGEN__::__v2_vec2_toMat2x2(v, v);
    __GEIGEN__::Matrix3x2d Fuu = __GEIGEN__::__M3x2_M2x2_Multiply(F, uu);
    __GEIGEN__::Matrix3x2d Fvv = __GEIGEN__::__M3x2_M2x2_Multiply(F, vv);

    double I6 = __GEIGEN__::__v_vec_dot(__M3x2_v2_multiply(F, u), __M3x2_v2_multiply(F, v));
    __GEIGEN__::Matrix2x2d uv = __GEIGEN__::__v2_vec2_toMat2x2(u, v);
    __GEIGEN__::Matrix2x2d vu = __GEIGEN__::__v2_vec2_toMat2x2(v, u);
    __GEIGEN__::Matrix3x2d Fuv = __GEIGEN__::__M3x2_M2x2_Multiply(F, uv);
    __GEIGEN__::Matrix3x2d Fvu = __GEIGEN__::__M3x2_M2x2_Multiply(F, vu);
    double utv = __GEIGEN__::__v2_vec_multiply(u, v);

    __GEIGEN__::Matrix3x2d PEPF_shear = __GEIGEN__::__S_Mat3x2_multiply(__GEIGEN__::__Mat3x2_add(Fuv, Fvu), 2 * (I6 - utv));
    __GEIGEN__::Matrix3x2d PEPF_stretch = __GEIGEN__::__Mat3x2_add(__GEIGEN__::__S_Mat3x2_multiply(Fuu, 2 * ucoeff), __GEIGEN__::__S_Mat3x2_multiply(Fvv, 2 * vcoeff));
    __GEIGEN__::Matrix3x2d PEPF = __GEIGEN__::__Mat3x2_add(__GEIGEN__::__S_Mat3x2_multiply(PEPF_shear, shearStiff), __GEIGEN__::__S_Mat3x2_multiply(PEPF_stretch, stretchStiff));
    return PEPF;
}
__device__
__GEIGEN__::Matrix3x3d __computePEPF_StableNHK3D_double(const __GEIGEN__::Matrix3x3d& F, const __GEIGEN__::Matrix3x3d& Sigma, const __GEIGEN__::Matrix3x3d& U, const __GEIGEN__::Matrix3x3d& V, double lengthRate, double volumRate) {

    double I3 = Sigma.m[0][0] * Sigma.m[1][1] * Sigma.m[2][2];
    double I2 = Sigma.m[0][0] * Sigma.m[0][0] + Sigma.m[1][1] * Sigma.m[1][1] + Sigma.m[2][2] * Sigma.m[2][2];

    double u = lengthRate, r = volumRate;
    __GEIGEN__::Matrix3x3d pI3pF;

    pI3pF.m[0][0] = F.m[1][1] * F.m[2][2] - F.m[1][2] * F.m[2][1];
    pI3pF.m[0][1] = F.m[1][2] * F.m[2][0] - F.m[1][0] * F.m[2][2];
    pI3pF.m[0][2] = F.m[1][0] * F.m[2][1] - F.m[1][1] * F.m[2][0];

    pI3pF.m[1][0] = F.m[2][1] * F.m[0][2] - F.m[2][2] * F.m[0][1];
    pI3pF.m[1][1] = F.m[2][2] * F.m[0][0] - F.m[2][0] * F.m[0][2];
    pI3pF.m[1][2] = F.m[2][0] * F.m[0][1] - F.m[2][1] * F.m[0][0];

    pI3pF.m[2][0] = F.m[0][1] * F.m[1][2] - F.m[1][1] * F.m[0][2];
    pI3pF.m[2][1] = F.m[0][2] * F.m[1][0] - F.m[0][0] * F.m[1][2];
    pI3pF.m[2][2] = F.m[0][0] * F.m[1][1] - F.m[0][1] * F.m[1][0];


    //printf("volRate and LenRate:  %f    %f\n", volumRate, lengthRate);

    __GEIGEN__::Matrix3x3d PEPF, tempA, tempB;
    tempA = __GEIGEN__::__S_Mat_multiply(F, u * (1 - 1 / (I2 + 1)));
    tempB = __GEIGEN__::__S_Mat_multiply(pI3pF, (r * (I3 - 1 - u * 3 / (r * 4))));
    __GEIGEN__::__Mat_add(tempA, tempB, PEPF);
    return PEPF;
}

__device__
__GEIGEN__::Matrix3x3d computePEPF_ARAP_double(const __GEIGEN__::Matrix3x3d& F, const __GEIGEN__::Matrix3x3d& Sigma, const __GEIGEN__::Matrix3x3d& U, const __GEIGEN__::Matrix3x3d& V, const double& lengthRate)
{
    __GEIGEN__::Matrix3x3d R, S;

    S = __GEIGEN__::__M_Mat_multiply(__GEIGEN__::__M_Mat_multiply(V, Sigma), __GEIGEN__::__Transpose3x3(V));//V * sigma * V.transpose();
    R = __GEIGEN__::__M_Mat_multiply(U, __GEIGEN__::__Transpose3x3(V));
    __GEIGEN__::Matrix3x3d g = __GEIGEN__::__Mat3x3_minus(F, R);
    return __GEIGEN__::__S_Mat_multiply(g, lengthRate);//lengthRate * g;
}

__device__
__GEIGEN__::Matrix9x9d project_ARAP_H_3D(const __GEIGEN__::Matrix3x3d& Sigma, const __GEIGEN__::Matrix3x3d& U, const __GEIGEN__::Matrix3x3d& V, const double& lengthRate)
{
    __GEIGEN__::Matrix3x3d R, S;

    S = __GEIGEN__::__M_Mat_multiply(__GEIGEN__::__M_Mat_multiply(V, Sigma), __GEIGEN__::__Transpose3x3(V));//V * sigma * V.transpose();
    R = __GEIGEN__::__M_Mat_multiply(U, __GEIGEN__::__Transpose3x3(V));
    __GEIGEN__::Matrix3x3d T0, T1, T2;

    __GEIGEN__::__set_Mat_val(T0, 0, -1, 0, 1, 0, 0, 0, 0, 0);
    __GEIGEN__::__set_Mat_val(T1, 0, 0, 0, 0, 0, 1, 0, -1, 0);
    __GEIGEN__::__set_Mat_val(T2, 0, 0, 1, 0, 0, 0, -1, 0, 0);

    double ml = 1 / sqrt(2.0);

    __GEIGEN__::Matrix3x3d VTransp = __GEIGEN__::__Transpose3x3(V);

    T0 = __GEIGEN__::__S_Mat_multiply(__GEIGEN__::__M_Mat_multiply(__GEIGEN__::__M_Mat_multiply(U, T0), VTransp), ml);
    T1 = __GEIGEN__::__S_Mat_multiply(__GEIGEN__::__M_Mat_multiply(__GEIGEN__::__M_Mat_multiply(U, T1), VTransp), ml);
    T2 = __GEIGEN__::__S_Mat_multiply(__GEIGEN__::__M_Mat_multiply(__GEIGEN__::__M_Mat_multiply(U, T2), VTransp), ml);

    __GEIGEN__::Vector9 t0, t1, t2;
    t0 = __GEIGEN__::__Mat3x3_to_vec9_double(T0);
    t1 = __GEIGEN__::__Mat3x3_to_vec9_double(T1);
    t2 = __GEIGEN__::__Mat3x3_to_vec9_double(T2);

    double sx = Sigma.m[0][0];
    double sy = Sigma.m[1][1];
    double sz = Sigma.m[2][2];
    double lambda0 = 2 / (sx + sy);
    double lambda1 = 2 / (sz + sy);
    double lambda2 = 2 / (sx + sz);

    if (sx + sy < 2)lambda0 = 1;
    if (sz + sy < 2)lambda1 = 1;
    if (sx + sz < 2)lambda2 = 1;

    __GEIGEN__::Matrix9x9d SH, M9_temp;
    __GEIGEN__::__identify_Mat9x9(SH);
    __GEIGEN__::Vector9 V9_temp;


    M9_temp = __GEIGEN__::__v9_vec9_toMat9x9(t0, t0);
    M9_temp = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, -lambda0);
    SH = __GEIGEN__::__Mat9x9_add(SH, M9_temp);

    M9_temp = __GEIGEN__::__v9_vec9_toMat9x9(t1, t1);
    M9_temp = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, -lambda1);
    SH = __GEIGEN__::__Mat9x9_add(SH, M9_temp);

    M9_temp = __GEIGEN__::__v9_vec9_toMat9x9(t2, t2);
    M9_temp = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, -lambda2);
    SH = __GEIGEN__::__Mat9x9_add(SH, M9_temp);

    return __GEIGEN__::__S_Mat9x9_multiply(SH, lengthRate);;
}
__device__
__GEIGEN__::Matrix6x6d __project_BaraffWitkinStretch_H(const __GEIGEN__::Matrix3x2d& F) {

    __GEIGEN__::Matrix6x6d H;
    H.m[0][0] = H.m[0][1] = H.m[0][2] = H.m[0][3] = H.m[0][4] = H.m[0][5] = 0;
    H.m[1][0] = H.m[1][1] = H.m[1][2] = H.m[1][3] = H.m[1][4] = H.m[1][5] = 0;
    H.m[2][0] = H.m[2][1] = H.m[2][2] = H.m[2][3] = H.m[2][4] = H.m[2][5] = 0;
    H.m[3][0] = H.m[3][1] = H.m[3][2] = H.m[3][3] = H.m[3][4] = H.m[3][5] = 0;
    H.m[4][0] = H.m[4][1] = H.m[4][2] = H.m[4][3] = H.m[4][4] = H.m[4][5] = 0;
    H.m[5][0] = H.m[5][1] = H.m[5][2] = H.m[5][3] = H.m[5][4] = H.m[5][5] = 0;
    double2 u, v;
    u.x = 1; u.y = 0;
    v.x = 0; v.y = 1;
    double I5u = __GEIGEN__::__squaredNorm(__M3x2_v2_multiply(F, u));
    double I5v = __GEIGEN__::__squaredNorm(__M3x2_v2_multiply(F, v));



    double invSqrtI5u = 1.0 / sqrt(I5u);
    double invSqrtI5v = 1.0 / sqrt(I5v);
    if (1 - invSqrtI5u > 0)
        H.m[0][0] = H.m[1][1] = H.m[2][2] = 2 * (1 - invSqrtI5u);
    if (1 - invSqrtI5v > 0)
        H.m[3][3] = H.m[4][4] = H.m[5][5] = 2 * (1 - invSqrtI5v);


    double uCoeff = (1.0 - invSqrtI5u >= 0.0) ? invSqrtI5u : 1.0;
    double vCoeff = (1.0 - invSqrtI5v >= 0.0) ? invSqrtI5v : 1.0;
    uCoeff *= 2; vCoeff *= 2;

    if (I5u < 1) {
        uCoeff *= 1e-2;
    }
    if (I5v < 1) {
        vCoeff *= 1e-2;
    }


    double3 fu, fv;
    fu.x = F.m[0][0]; fu.y = F.m[1][0]; fu.z = F.m[2][0];
    fv.x = F.m[0][1]; fv.y = F.m[1][1]; fv.z = F.m[2][1];
    fu = __GEIGEN__::__normalized(fu);
    fv = __GEIGEN__::__normalized(fv);

    __GEIGEN__::Matrix3x3d cfufu = __GEIGEN__::__S_Mat_multiply(__GEIGEN__::__v_vec_toMat(fu, fu), uCoeff);
    __GEIGEN__::Matrix3x3d cfvfv = __GEIGEN__::__S_Mat_multiply(__GEIGEN__::__v_vec_toMat(fv, fv), vCoeff);
    H.m[0][0] += cfufu.m[0][0]; H.m[0][1] += cfufu.m[0][1]; H.m[0][2] += cfufu.m[0][2];
    H.m[1][0] += cfufu.m[1][0]; H.m[1][1] += cfufu.m[1][1]; H.m[1][2] += cfufu.m[1][2];
    H.m[2][0] += cfufu.m[2][0]; H.m[2][1] += cfufu.m[2][1]; H.m[2][2] += cfufu.m[2][2];


    H.m[3][3] += cfvfv.m[0][0]; H.m[3][4] += cfvfv.m[0][1]; H.m[3][5] += cfvfv.m[0][2];
    H.m[4][3] += cfvfv.m[1][0]; H.m[4][4] += cfvfv.m[1][1]; H.m[4][5] += cfvfv.m[1][2];
    H.m[5][3] += cfvfv.m[2][0]; H.m[5][4] += cfvfv.m[2][1]; H.m[5][5] += cfvfv.m[2][2];
    return H;
}


__device__
__GEIGEN__::Matrix6x6d __project_BaraffWitkinShear_H(const __GEIGEN__::Matrix3x2d& F) {

    __GEIGEN__::Matrix6x6d H;
    H.m[0][0] = H.m[0][1] = H.m[0][2] = H.m[0][3] = H.m[0][4] = H.m[0][5] = 0;
    H.m[1][0] = H.m[1][1] = H.m[1][2] = H.m[1][3] = H.m[1][4] = H.m[1][5] = 0;
    H.m[2][0] = H.m[2][1] = H.m[2][2] = H.m[2][3] = H.m[2][4] = H.m[2][5] = 0;
    H.m[3][0] = H.m[3][1] = H.m[3][2] = H.m[3][3] = H.m[3][4] = H.m[3][5] = 0;
    H.m[4][0] = H.m[4][1] = H.m[4][2] = H.m[4][3] = H.m[4][4] = H.m[4][5] = 0;
    H.m[5][0] = H.m[5][1] = H.m[5][2] = H.m[5][3] = H.m[5][4] = H.m[5][5] = 0;
    double2 u, v;
    u.x = 1; u.y = 0;
    v.x = 0; v.y = 1;
    __GEIGEN__::Matrix6x6d H_shear;

    H_shear.m[0][0] = H_shear.m[0][1] = H_shear.m[0][2] = H_shear.m[0][3] = H_shear.m[0][4] = H_shear.m[0][5] = 0;
    H_shear.m[1][0] = H_shear.m[1][1] = H_shear.m[1][2] = H_shear.m[1][3] = H_shear.m[1][4] = H_shear.m[1][5] = 0;
    H_shear.m[2][0] = H_shear.m[2][1] = H_shear.m[2][2] = H_shear.m[2][3] = H_shear.m[2][4] = H_shear.m[2][5] = 0;
    H_shear.m[3][0] = H_shear.m[3][1] = H_shear.m[3][2] = H_shear.m[3][3] = H_shear.m[3][4] = H_shear.m[3][5] = 0;
    H_shear.m[4][0] = H_shear.m[4][1] = H_shear.m[4][2] = H_shear.m[4][3] = H_shear.m[4][4] = H_shear.m[4][5] = 0;
    H_shear.m[5][0] = H_shear.m[5][1] = H_shear.m[5][2] = H_shear.m[5][3] = H_shear.m[5][4] = H_shear.m[5][5] = 0;
    H_shear.m[3][0] = H_shear.m[4][1] =
    H_shear.m[5][2] = H_shear.m[0][3] =
    H_shear.m[1][4] = H_shear.m[2][5] = 1.0;
    double I6 = __GEIGEN__::__v_vec_dot(__M3x2_v2_multiply(F, u), __M3x2_v2_multiply(F, v));
    double signI6 = (I6 >= 0) ? 1.0 : -1.0;

    __GEIGEN__::Matrix2x2d uv = __GEIGEN__::__v2_vec2_toMat2x2(u, v);
    __GEIGEN__::Matrix2x2d vu = __GEIGEN__::__v2_vec2_toMat2x2(v, u);
    __GEIGEN__::Matrix3x2d Fuv = __GEIGEN__::__M3x2_M2x2_Multiply(F, uv);
    __GEIGEN__::Matrix3x2d Fvu = __GEIGEN__::__M3x2_M2x2_Multiply(F, vu);
    __GEIGEN__::Vector6 g;
    g.v[0] = Fuv.m[0][0] + Fvu.m[0][0];
    g.v[1] = Fuv.m[1][0] + Fvu.m[1][0];
    g.v[2] = Fuv.m[2][0] + Fvu.m[2][0];
    g.v[3] = Fuv.m[0][1] + Fvu.m[0][1];
    g.v[4] = Fuv.m[1][1] + Fvu.m[1][1];
    g.v[5] = Fuv.m[2][1] + Fvu.m[2][1];
    double I2 = F.m[0][0] * F.m[0][0] + F.m[0][1] * F.m[0][1] + F.m[1][0] * F.m[1][0] + F.m[1][1] * F.m[1][1] + F.m[2][0] * F.m[2][0] + F.m[2][1] * F.m[2][1];
    double lambda0 = 0.5 * (I2 + sqrt(I2 * I2 + 12 * I6 * I6));
    __GEIGEN__::Vector6 q0 = __GEIGEN__::__M6x6_v6_multiply(H_shear, g);
    q0 = __GEIGEN__::__s_vec6_multiply(q0, I6);
    q0 = __GEIGEN__::__add6(q0, __s_vec6_multiply(g, lambda0));
    __GEIGEN__::__normalized_vec6_double(q0);
    __GEIGEN__::Matrix6x6d T;
    T.m[0][0] = T.m[0][1] = T.m[0][2] = T.m[0][3] = T.m[0][4] = T.m[0][5] = 0;
    T.m[1][0] = T.m[1][1] = T.m[1][2] = T.m[1][3] = T.m[1][4] = T.m[1][5] = 0;
    T.m[2][0] = T.m[2][1] = T.m[2][2] = T.m[2][3] = T.m[2][4] = T.m[2][5] = 0;
    T.m[3][0] = T.m[3][1] = T.m[3][2] = T.m[3][3] = T.m[3][4] = T.m[3][5] = 0;
    T.m[4][0] = T.m[4][1] = T.m[4][2] = T.m[4][3] = T.m[4][4] = T.m[4][5] = 0;
    T.m[5][0] = T.m[5][1] = T.m[5][2] = T.m[5][3] = T.m[5][4] = T.m[5][5] = 0;
    T.m[0][0] = T.m[1][1] =
    T.m[2][2] = T.m[3][3] =
    T.m[4][4] = T.m[5][5] = 1.0;
    __GEIGEN__::__Mat_add(T, __S_Mat6x6_multiply(H_shear, signI6), T);
    T = __GEIGEN__::__S_Mat6x6_multiply(T, 0.5);
    __GEIGEN__::Vector6 Tq = __GEIGEN__::__M6x6_v6_multiply(T, q0);
    double normTQ = Tq.v[0] * Tq.v[0] + Tq.v[1] * Tq.v[1] + Tq.v[2] * Tq.v[2] + Tq.v[3] * Tq.v[3] + Tq.v[4] * Tq.v[4] + Tq.v[5] * Tq.v[5];

    __GEIGEN__::Matrix6x6d Tmp;
    __GEIGEN__::__Mat_add(T, __GEIGEN__::__S_Mat6x6_multiply(__v6_vec6_toMat6x6(Tq, Tq), -1.0 / normTQ), Tmp);
    Tmp = __S_Mat6x6_multiply(Tmp, fabs(I6));
    __GEIGEN__::__Mat_add(Tmp, __S_Mat6x6_multiply(__v6_vec6_toMat6x6(q0, q0), lambda0), Tmp);
    Tmp = __S_Mat6x6_multiply(Tmp, 2);
    return Tmp;
}

__device__ 
__GEIGEN__::Matrix9x9d __project_StabbleNHK_H_3D(const double3& sigma, const __GEIGEN__::Matrix3x3d& U, const __GEIGEN__::Matrix3x3d& V, const double& lengthRate, const double& volumRate, __GEIGEN__::Matrix9x9d& H) {
    double sigxx = sigma.x * sigma.x;
    double sigyy = sigma.y * sigma.y;
    double sigzz = sigma.z * sigma.z;

    double I3 = sigma.x * sigma.y * sigma.z;
    double I2 = sigxx + sigyy + sigzz;
    double g2 = sigxx*sigyy + sigxx*sigzz + sigyy*sigzz;

    double u = lengthRate, r = volumRate;

    double n = 2 * u / ((I2 + 1) * (I2 + 1) * (r * (I3 - 1) - 3 * u / 4));
    double p = r / (r * (I3 - 1) - 3 * u / 4);
    double c2 = -g2 * p - I2 * n;
    double c1 = -(1 + 2 * I3 * p) * I2 - 6 * I3 * n + (g2 * I2 - 9 * I3 * I3) * p * n;
    double c0 = -(2 + 3 * I3 * p) * I3 + (I2 * I2 - 4 * g2) * n + 2 * I3 * p * n * (I2 * I2 - 3 * g2);

    double roots[3] = { 0 };
    int num_solution = 0;
    __GEIGEN__::__NewtonSolverForCubicEquation_satbleNeohook(1, c2, c1, c0, roots, num_solution, 1e-6);

    __GEIGEN__::Matrix3x3d D[3], M_temp[3];
    double q[3];
    __GEIGEN__::Matrix3x3d Q[9];
    double lamda[9];
    double Ut = u * (1 - 1 / (I2 + 1));
    double alpha = 1 + 3 * u / r / 4;

    double I3minuAlphaDotR = (I3 - alpha)*r;

    for (int i = 0; i < num_solution; i++) {
        double alpha0 = roots[i] * (sigma.y + sigma.x * sigma.z * n + I3 * sigma.y * p) +
                        sigma.x * sigma.z + sigma.y * (sigxx - sigyy + sigzz) * n +
                        I3 * sigma.x * sigma.z * p +
                        sigma.x * (sigxx - sigyy) * sigma.z *
                        (sigyy - sigzz) * p * n;

        double alpha1 = roots[i] * (sigma.x + sigma.y * sigma.z * n + I3 * sigma.x * p) +
                        sigma.y * sigma.z - sigma.x * (sigxx - sigyy - sigzz) * n +
                        I3 * sigma.y * sigma.z * p -
                        sigma.y * (sigxx - sigyy) * sigma.z *
                        (sigxx - sigzz) * p * n;

        double alpha2 = roots[i] * roots[i] - roots[i] * (sigxx + sigyy) * (n + sigzz * p) -
                        sigzz - 2 * I3 * n - 2 * I3 * sigzz * p +
                        ((sigxx - sigyy) * sigma.z) * ((sigxx - sigyy) * sigma.z) * p * n;

        double normalSum = alpha0 * alpha0 + alpha1 * alpha1 + alpha2 * alpha2;

        if (normalSum == 0) {
            lamda[i] = 0; continue;
        }

        q[i] = 1 / sqrt(normalSum);
        __GEIGEN__::__set_Mat_val(D[i], alpha0, 0, 0, 0, alpha1, 0, 0, 0, alpha2);

        __GEIGEN__::__s_M_Mat_MT_multiply(U, D[i], V, q[i], Q[i]);
        lamda[i] = I3minuAlphaDotR * roots[i] + Ut;
    }


    lamda[3] = Ut + sigma.z * I3minuAlphaDotR;
    lamda[4] = Ut + sigma.x * I3minuAlphaDotR;
    lamda[5] = Ut + sigma.y * I3minuAlphaDotR;

    lamda[6] = Ut - sigma.z * I3minuAlphaDotR;
    lamda[7] = Ut - sigma.x * I3minuAlphaDotR;
    lamda[8] = Ut - sigma.y * I3minuAlphaDotR;

    __GEIGEN__::__set_Mat_val(Q[3], 0, -1, 0, 1, 0, 0, 0, 0, 0);
    __GEIGEN__::__set_Mat_val(Q[4], 0, 0, 0, 0, 0, 1, 0, -1, 0);
    __GEIGEN__::__set_Mat_val(Q[5], 0, 0, 1, 0, 0, 0, -1, 0, 0);
    __GEIGEN__::__set_Mat_val(Q[6], 0, 1, 0, 1, 0, 0, 0, 0, 0);
    __GEIGEN__::__set_Mat_val(Q[7], 0, 0, 0, 0, 0, 1, 0, 1, 0);
    __GEIGEN__::__set_Mat_val(Q[8], 0, 0, 1, 0, 0, 0, 1, 0, 0);

    double ml = 1 / sqrt(2.0);

    M_temp[1] = __GEIGEN__::__Transpose3x3(V);
    for (int i = 3;i < 9;i++) {
        __GEIGEN__::__M_Mat_multiply(U, Q[i], M_temp[0]);
        __GEIGEN__::__M_Mat_multiply(M_temp[0], M_temp[1], M_temp[2]);
        Q[i] = __GEIGEN__::__S_Mat_multiply(M_temp[2], ml);

        //Q[i] = __GEIGEN__::__s_M_Mat_MT_multiply(U, Q[i], V, ml);
    }

    __GEIGEN__::Matrix9x9d M9_temp;
    __GEIGEN__::__init_Mat9x9(H, 0);
    __GEIGEN__::Vector9 V9_temp;
    for (int i = 0; i < 9; i++) {
        if (lamda[i] > 0) {
            V9_temp = __GEIGEN__::__Mat3x3_to_vec9_double(Q[i]);
            M9_temp = __GEIGEN__::__v9_vec9_toMat9x9(V9_temp, V9_temp, lamda[i]);
            H = __GEIGEN__::__Mat9x9_add(H, M9_temp);
        }
    }
}



__device__ 
__GEIGEN__::Matrix9x9d __project_ANIOSI5_H_3D(const __GEIGEN__::Matrix3x3d& F, const __GEIGEN__::Matrix3x3d& sigma, const __GEIGEN__::Matrix3x3d& U, const __GEIGEN__::Matrix3x3d& V, const double3& fiber_direction, const double& scale, const double& contract_length) {
    double3 direction = __GEIGEN__::__normalized(fiber_direction);
    __GEIGEN__::Matrix3x3d S, M_temp[3], Vtranspose;


    //S = V * sigma * V.transpose();
    __GEIGEN__::__M_Mat_multiply(V, sigma, M_temp[0]);
    Vtranspose = __GEIGEN__::__Transpose3x3(V);
    __GEIGEN__::__M_Mat_multiply(M_temp[0], Vtranspose, S);
    //__S_Mat_multiply(M_temp[2], ml, Q[i]);

    double3 v_temp = __GEIGEN__::__M_v_multiply(S, direction);
    double I4 = __GEIGEN__::__v_vec_dot(direction, v_temp);//direction.transpose() * S * direction;
    double I5 = __GEIGEN__::__v_vec_dot(v_temp, v_temp);//direction.transpose() * S.transpose() * S * direction;

    __GEIGEN__::Matrix9x9d H;
    __GEIGEN__::__init_Mat9x9(H, 0);
    if (abs(I5) < 1e-15) return H;

    double s = 0;
    if (I4 < 0) {
        s = -1;
    }
    else if (I4 > 0) {
        s = 1;
    }

    double lamda0 = scale;
    double lamda1 = scale * (1 - s * contract_length / sqrt(I5));
    double lamda2 = lamda1;
    //double lamda2 = lamda1;
    __GEIGEN__::Matrix3x3d Q0, Q1, Q2, A;
    A = __GEIGEN__::__v_vec_toMat(direction, direction);

    __GEIGEN__::__M_Mat_multiply(F, A, M_temp[0]);
    Q0 = __GEIGEN__::__S_Mat_multiply(M_temp[0], (1 / sqrt(I5)));
    //Q0 = (1 / sqrt(I5)) * F * A;

    __GEIGEN__::Matrix3x3d Tx, Ty, Tz;

    __GEIGEN__::__set_Mat_val(Tx, 0, 0, 0, 0, 0, 1, 0, -1, 0);
    __GEIGEN__::__set_Mat_val(Ty, 0, 0, -1, 0, 0, 0, 1, 0, 0);
    __GEIGEN__::__set_Mat_val(Tz, 0, 1, 0, -1, 0, 0, 0, 0, 0);

    //__Transpose3x3(V, M_temp[0]);
    double3 directionM = __GEIGEN__::__M_v_multiply(Vtranspose, direction);

    double ratio = 1.f / sqrt(2.f);
    Tx = __GEIGEN__::__S_Mat_multiply(Tx, ratio);
    Ty = __GEIGEN__::__S_Mat_multiply(Ty, ratio);
    Tz = __GEIGEN__::__S_Mat_multiply(Tz, ratio);

    //Q1 = U * Tx * sigma * V.transpose() * A;
    __GEIGEN__::__M_Mat_multiply(U, Tx, M_temp[1]);
    __GEIGEN__::__M_Mat_multiply(M_temp[1], sigma, M_temp[2]);
    __GEIGEN__::__M_Mat_multiply(M_temp[2], Vtranspose, M_temp[1]);
    __GEIGEN__::__M_Mat_multiply(M_temp[1], A, Q1);

    //Q2 = (sigma(1, 1) * directionM[1]) * U * Tz * sigma * V.transpose() * A - (sigma(2, 2) * directionM[2]) * U * Ty * sigma * V.transpose() * A;
    __GEIGEN__::__M_Mat_multiply(U, Tz, M_temp[0]);
    __GEIGEN__::__M_Mat_multiply(M_temp[0], sigma, M_temp[1]);
    __GEIGEN__::__M_Mat_multiply(M_temp[1], Vtranspose, M_temp[2]);
    __GEIGEN__::__M_Mat_multiply(M_temp[2], A, M_temp[0]);
    M_temp[0] = __S_Mat_multiply(M_temp[0], (sigma.m[1][1] * directionM.y));
    __GEIGEN__::__M_Mat_multiply(U, Ty, M_temp[1]);
    __GEIGEN__::__M_Mat_multiply(M_temp[1], sigma, M_temp[2]);
    __GEIGEN__::__M_Mat_multiply(M_temp[2], Vtranspose, M_temp[1]);
    __GEIGEN__::__M_Mat_multiply(M_temp[1], A, M_temp[2]);
    M_temp[2] = __GEIGEN__::__S_Mat_multiply(M_temp[2], -(sigma.m[2][2] * directionM.z));
    __GEIGEN__::__Mat_add(M_temp[0], M_temp[2], Q2);

    //H = lamda0 * vec_double(Q0) * vec_double(Q0).transpose();
    __GEIGEN__::Vector9 V9_temp = __GEIGEN__::__Mat3x3_to_vec9_double(Q0);
    __GEIGEN__::Matrix9x9d M9_temp = __GEIGEN__::__v9_vec9_toMat9x9(V9_temp, V9_temp);
    H = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lamda0);
    //H = __Mat9x9_add(H, M9_temp[1]);
    if (lamda1 > 0) {
        //H += lamda1 * vec_double(Q1) * vec_double(Q1).transpose();
        //H += lamda2 * vec_double(Q2) * vec_double(Q2).transpose();
        V9_temp = __GEIGEN__::__Mat3x3_to_vec9_double(Q1);
        M9_temp = __GEIGEN__::__v9_vec9_toMat9x9(V9_temp, V9_temp);
        M9_temp = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lamda1);
        H = __GEIGEN__::__Mat9x9_add(H, M9_temp);

        V9_temp = __GEIGEN__::__Mat3x3_to_vec9_double(Q2);
        M9_temp = __GEIGEN__::__v9_vec9_toMat9x9(V9_temp, V9_temp);
        M9_temp = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lamda2);
        H = __GEIGEN__::__Mat9x9_add(H, M9_temp);
    }

    return H;
}

__device__ 
__GEIGEN__::Matrix3x3d __computePEPF_Aniostropic3D_double(const __GEIGEN__::Matrix3x3d& F, double3 fiber_direction, const double& scale, const double& contract_length) {

    double3 direction = __GEIGEN__::__normalized(fiber_direction);
    __GEIGEN__::Matrix3x3d U, V, S, sigma, M_Temp0, M_Temp1;
    SVD(F, U, V, sigma);
    __GEIGEN__::__M_Mat_multiply(V, sigma, M_Temp0);
    M_Temp1 = __GEIGEN__::__Transpose3x3(V);
    __GEIGEN__::__M_Mat_multiply(M_Temp0, M_Temp1, S);
    double3 V_Temp0, V_Temp1;
    V_Temp0 = __GEIGEN__::__v_M_multiply(direction, S);
    double I4, I5;
    I4 = __GEIGEN__::__v_vec_dot(V_Temp0, direction);
    V_Temp1 = __GEIGEN__::__M_v_multiply(S, direction);
    I5 = __GEIGEN__::__v_vec_dot(V_Temp1, V_Temp1);

    if (I4 == 0) {
        // system("pause");
    }

    double s = 0;
    if (I4 < 0) {
        s = -1;
    }
    else if (I4 > 0) {
        s = 1;
    }

    __GEIGEN__::Matrix3x3d PEPF;
    double s_temp0 = scale * (1 - s * contract_length / sqrt(I5));
    M_Temp0 = __GEIGEN__::__v_vec_toMat(direction, direction);
    __GEIGEN__::__M_Mat_multiply(F, M_Temp0, M_Temp1);
    PEPF = __GEIGEN__::__S_Mat_multiply(M_Temp1, s_temp0);
    return PEPF;
}
__device__
double __cal_BaraffWitkinStretch_energy(const double3* vertexes, const uint3& triangle, const __GEIGEN__::Matrix2x2d& triDmInverse, const double& area, const double& stretchStiff, const double& shearhStiff) {
    __GEIGEN__::Matrix3x2d Ds;
    __calculateDs2D_double(vertexes, triangle, Ds);
    __GEIGEN__::Matrix3x2d F = __GEIGEN__::__M3x2_M2x2_Multiply(Ds, triDmInverse);

    double2 u, v;
    u.x = 1; u.y = 0;
    v.x = 0; v.y = 1;
    double I5u = __GEIGEN__::__squaredNorm(__M3x2_v2_multiply(F, u));
    double I5v = __GEIGEN__::__squaredNorm(__M3x2_v2_multiply(F, v));

    double ucoeff = 1;
    double vcoeff = 1;
    if (I5u < 1) {
        ucoeff *= 1e-2;
    }
    if (I5v < 1) {
        vcoeff *= 1e-2;
    }


    double I6 = __GEIGEN__::__v_vec_dot(__M3x2_v2_multiply(F, u), __M3x2_v2_multiply(F, v));
    return area * (stretchStiff * (ucoeff * pow(sqrt(I5u) - 1, 2) + vcoeff * pow(sqrt(I5v) - 1, 2)) + shearhStiff * I6 * I6);
}
__device__
double __cal_StabbleNHK_energy_3D(const double3* vertexes, const uint4& tetrahedra, const __GEIGEN__::Matrix3x3d& DmInverse, const double& volume, const double& lenRate, const double& volRate) {
    __GEIGEN__::Matrix3x3d Ds;
    __calculateDms3D_double(vertexes, tetrahedra, Ds);
    __GEIGEN__::Matrix3x3d F;
    __M_Mat_multiply(Ds, DmInverse, F);
    //printf("%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n\n\n\n\n\n", F.m[0][0], F.m[0][1], F.m[0][2], F.m[1][0], F.m[1][1], F.m[1][2], F.m[2][0], F.m[2][1], F.m[2][2]);
    __GEIGEN__::Matrix3x3d U, V, sigma, S;
    __GEIGEN__::Matrix3x3d M_Temp0, M_Temp1;
    SVD(F, U, V, sigma);
    //printf("%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n\n\n\n\n\n", V.m[0][0], V.m[0][1], V.m[0][2], V.m[1][0], V.m[1][1], V.m[1][2], V.m[2][0], V.m[2][1], V.m[2][2]);
    //printf("%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n\n\n\n\n\n", U.m[0][0], U.m[0][1], U.m[0][2], U.m[1][0], U.m[1][1], U.m[1][2], U.m[2][0], U.m[2][1], U.m[2][2]);
    __GEIGEN__::__M_Mat_multiply(V, sigma, M_Temp0);
    M_Temp1 = __GEIGEN__::__Transpose3x3(V);
    __GEIGEN__::__M_Mat_multiply(M_Temp0, M_Temp1, S);

    __GEIGEN__::__M_Mat_multiply(S, S, M_Temp0);
    double I2 = __GEIGEN__::__Mat_Trace(M_Temp0);
    double I3;
    __GEIGEN__::__Determiant(S, I3);
    //printf("%f     %f\n\n\n", I2, I3);
    return (0.5 * lenRate * (I2 - 3) + 0.5 * volRate * (I3 - 1 - 3 * lenRate / 4 / volRate) * (I3 - 1 - 3 * lenRate / 4 / volRate) - 0.5 * lenRate * log(I2 + 1) /*- (0.5 * volRate * (3 * lenRate / 4 / volRate) * (3 * lenRate / 4 / volRate) - 0.5 * lenRate * log(4.0))*/) * volume;
    //printf("I2   I3   ler  volr\n", I2, I3, lenRate, volRate);
}

__device__
double __cal_ARAP_energy_3D(const double3* vertexes, const uint4& tetrahedra, const __GEIGEN__::Matrix3x3d& DmInverse, const double& volume, const double& lenRate) {
    __GEIGEN__::Matrix3x3d Ds;
    __calculateDms3D_double(vertexes, tetrahedra, Ds);
    __GEIGEN__::Matrix3x3d F;
    __M_Mat_multiply(Ds, DmInverse, F);
    //printf("%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n\n\n\n\n\n", F.m[0][0], F.m[0][1], F.m[0][2], F.m[1][0], F.m[1][1], F.m[1][2], F.m[2][0], F.m[2][1], F.m[2][2]);
    __GEIGEN__::Matrix3x3d U, V, sigma, S, R;
    __GEIGEN__::Matrix3x3d M_Temp0, M_Temp1;
    SVD(F, U, V, sigma);

    S = __GEIGEN__::__M_Mat_multiply(__GEIGEN__::__M_Mat_multiply(V, sigma), __GEIGEN__::__Transpose3x3(V));//V * sigma * V.transpose();
    R = __GEIGEN__::__M_Mat_multiply(U, __GEIGEN__::__Transpose3x3(V));
    __GEIGEN__::Matrix3x3d g = __GEIGEN__::__Mat3x3_minus(F, R);
    double energy = 0;
    for (int i = 0;i < 3;i++) {
        for (int j = 0;j < 3;j++) {
            energy += g.m[i][j] * g.m[i][j];
        }
    }
    return energy * volume * lenRate * 0.5;
    //printf("I2   I3   ler  volr\n", I2, I3, lenRate, volRate);
}

__device__
__GEIGEN__::Matrix9x12d __computePFDsPX3D_double(const __GEIGEN__::Matrix3x3d& InverseDm) {
    __GEIGEN__::Matrix9x12d matOut;
    __GEIGEN__::__init_Mat9x12_val(matOut, 0);
    double m = InverseDm.m[0][0], n = InverseDm.m[0][1], o = InverseDm.m[0][2];
    double p = InverseDm.m[1][0], q = InverseDm.m[1][1], r = InverseDm.m[1][2];
    double s = InverseDm.m[2][0], t = InverseDm.m[2][1], u = InverseDm.m[2][2];
    double t1 = -(m + p + s);
    double t2 = -(n + q + t);
    double t3 = -(o + r + u);
    matOut.m[0][0] = t1;  matOut.m[0][3] = m;  matOut.m[0][6] = p;  matOut.m[0][9] = s;
    matOut.m[1][1] = t1;  matOut.m[1][4] = m;  matOut.m[1][7] = p;  matOut.m[1][10] = s;
    matOut.m[2][2] = t1;  matOut.m[2][5] = m;  matOut.m[2][8] = p;  matOut.m[2][11] = s;
    matOut.m[3][0] = t2;  matOut.m[3][3] = n;  matOut.m[3][6] = q;  matOut.m[3][9] = t;
    matOut.m[4][1] = t2;  matOut.m[4][4] = n;  matOut.m[4][7] = q;  matOut.m[4][10] = t;
    matOut.m[5][2] = t2;  matOut.m[5][5] = n;  matOut.m[5][8] = q;  matOut.m[5][11] = t;
    matOut.m[6][0] = t3;  matOut.m[6][3] = o;  matOut.m[6][6] = r;  matOut.m[6][9] = u;
    matOut.m[7][1] = t3;  matOut.m[7][4] = o;  matOut.m[7][7] = r;  matOut.m[7][10] = u;
    matOut.m[8][2] = t3;  matOut.m[8][5] = o;  matOut.m[8][8] = r;  matOut.m[8][11] = u;

    return matOut;
}

__device__
__GEIGEN__::Matrix6x12d __computePFDsPX3D_6x12_double(const __GEIGEN__::Matrix2x2d& InverseDm) {
    __GEIGEN__::Matrix6x12d matOut;
    __GEIGEN__::__init_Mat6x12_val(matOut, 0);
    double m = InverseDm.m[0][0], n = InverseDm.m[0][1];
    double p = InverseDm.m[1][0], q = InverseDm.m[1][1];

    matOut.m[0][0] = -m; matOut.m[3][0] = -n;
    matOut.m[1][1] = -m; matOut.m[4][1] = -n;
    matOut.m[2][2] = -m; matOut.m[5][2] = -n;

    matOut.m[0][3] = -p; matOut.m[3][3] = -q;
    matOut.m[1][4] = -p; matOut.m[4][4] = -q;
    matOut.m[2][5] = -p; matOut.m[5][5] = -q;

    matOut.m[0][6] = p; matOut.m[3][6] = q;
    matOut.m[1][7] = p; matOut.m[4][7] = q;
    matOut.m[2][8] = p; matOut.m[5][8] = q;

    matOut.m[0][9] = m; matOut.m[3][9] = n;
    matOut.m[1][10] = m; matOut.m[4][10] = n;
    matOut.m[2][11] = m; matOut.m[5][11] = n;

    return matOut;
}

__device__
__GEIGEN__::Matrix6x9d __computePFDsPX3D_6x9_double(const __GEIGEN__::Matrix2x2d& InverseDm) {
    __GEIGEN__::Matrix6x9d matOut;
    __GEIGEN__::__init_Mat6x9_val(matOut, 0);
    double d0 = InverseDm.m[0][0], d2 = InverseDm.m[0][1];
    double d1 = InverseDm.m[1][0], d3 = InverseDm.m[1][1];

    double s0 = d0 + d1;
    double s1 = d2 + d3;

    matOut.m[0][0] = -s0;
    matOut.m[3][0] = -s1;

    // dF / dy0
    matOut.m[1][1] = -s0;
    matOut.m[4][1] = -s1;

    // dF / dz0
    matOut.m[2][2] = -s0;
    matOut.m[5][2] = -s1;

    // dF / dx1
    matOut.m[0][3] = d0;
    matOut.m[3][3] = d2;

    // dF / dy1
    matOut.m[1][4] = d0;
    matOut.m[4][4] = d2;

    // dF / dz1
    matOut.m[2][5] = d0;
    matOut.m[5][5] = d2;

    // dF / dx2
    matOut.m[0][6] = d1;
    matOut.m[3][6] = d3;

    // dF / dy2
    matOut.m[1][7] = d1;
    matOut.m[4][7] = d3;

    // dF / dz2
    matOut.m[2][8] = d1;
    matOut.m[5][8] = d3;

    return matOut;
}

__device__
__GEIGEN__::Matrix3x6d __computePFDsPX3D_3x6_double(const double& InverseDm) {
    __GEIGEN__::Matrix3x6d matOut;
    __GEIGEN__::__init_Mat3x6_val(matOut, 0);

    matOut.m[0][0] = -InverseDm;
    matOut.m[1][1] = -InverseDm;
    matOut.m[2][2] = -InverseDm;

    matOut.m[0][3] = InverseDm;
    matOut.m[1][4] = InverseDm;
    matOut.m[2][5] = InverseDm;

    return matOut;
}

__device__
__GEIGEN__::Matrix9x12d __computePFDmPX3D_double(const __GEIGEN__::Matrix12x9d& PDmPx, const __GEIGEN__::Matrix3x3d& Ds, const __GEIGEN__::Matrix3x3d& DmInv) {
    __GEIGEN__::Matrix9x12d DsPDminvPx;
    __GEIGEN__::__init_Mat9x12_val(DsPDminvPx, 0);

    for (int i = 0; i < 12; i++) {
        __GEIGEN__::Matrix3x3d PDmPxi = __GEIGEN__::__vec9_to_Mat3x3_double(PDmPx.m[i]);
        __GEIGEN__::Matrix3x3d DsPDminvPxi;
        __GEIGEN__::__M_Mat_multiply(Ds, __GEIGEN__::__M_Mat_multiply(__GEIGEN__::__M_Mat_multiply(DmInv, PDmPxi), DmInv), DsPDminvPxi);

        __GEIGEN__::Vector9 tmp = __GEIGEN__::__Mat3x3_to_vec9_double(DsPDminvPxi);

        for (int j = 0;j < 9;j++) {
            DsPDminvPx.m[j][i] = -tmp.v[j];
        }
    }

    return DsPDminvPx;
}

__device__
__GEIGEN__::Matrix6x12d __computePFDmPX3D_6x12_double(const __GEIGEN__::Matrix12x4d& PDmPx, const __GEIGEN__::Matrix3x2d& Ds, const __GEIGEN__::Matrix2x2d& DmInv) {
    __GEIGEN__::Matrix6x12d DsPDminvPx;
    __GEIGEN__::__init_Mat6x12_val(DsPDminvPx, 0);

    for (int i = 0; i < 12; i++) {
        __GEIGEN__::Matrix2x2d PDmPxi = __GEIGEN__::__vec4_to_Mat2x2_double(PDmPx.m[i]);

        __GEIGEN__::Matrix3x2d DsPDminvPxi = __GEIGEN__::__M3x2_M2x2_Multiply(Ds, __GEIGEN__::__M2x2_Mat2x2_multiply(__GEIGEN__::__M2x2_Mat2x2_multiply(DmInv, PDmPxi), DmInv));

        __GEIGEN__::Vector6 tmp = __GEIGEN__::__Mat3x2_to_vec6_double(DsPDminvPxi);
        for (int j = 0;j < 6;j++) {
            DsPDminvPx.m[j][i] = -tmp.v[j];
        }

    }

    return DsPDminvPx;
}

__device__
__GEIGEN__::Matrix3x6d __computePFDmPX3D_3x6_double(const __GEIGEN__::Vector6& PDmPx, const double3& Ds, const double& DmInv) {
    __GEIGEN__::Matrix3x6d DsPDminvPx;
    __GEIGEN__::__init_Mat3x6_val(DsPDminvPx, 0);

    for (int i = 0; i < 6; i++) {
        double PDmPxi = PDmPx.v[i];

        double3 DsPDminvPxi = __GEIGEN__::__s_vec_multiply(Ds, ((DmInv * PDmPxi) * DmInv));
        DsPDminvPx.m[0][i] = -DsPDminvPxi.x;
        DsPDminvPx.m[1][i] = -DsPDminvPxi.y;
        DsPDminvPx.m[2][i] = -DsPDminvPxi.z;
    }

    return DsPDminvPx;
}

template <typename Scalar, int size>
__device__ void PDSNK(Eigen::Matrix<Scalar, size, size>& symMtr)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, size, size>> eigenSolver(symMtr);
    if(eigenSolver.eigenvalues()[0] >= 0.0)
    {
        return;
    }
    Eigen::Matrix<Scalar, size, size> D;
    D.setZero();
    int rows = size;  //((size == Eigen::Dynamic) ? symMtr.rows() : size);
    for(int i = 0; i < rows; i++)
    {
        if(eigenSolver.eigenvalues()[i] > 0.0)
        {
            D(i, i) = eigenSolver.eigenvalues()[i];
        }
    }
    symMtr = eigenSolver.eigenvectors() * D * eigenSolver.eigenvectors().transpose();
}

__device__
__GEIGEN__::Matrix6x9d __computePFDmPX3D_6x9_double(const __GEIGEN__::Matrix9x4d& PDmPx, const __GEIGEN__::Matrix3x2d& Ds, const __GEIGEN__::Matrix2x2d& DmInv) {
    __GEIGEN__::Matrix6x9d DsPDminvPx;
    __GEIGEN__::__init_Mat6x9_val(DsPDminvPx, 0);

    for (int i = 0; i < 9; i++) {
        __GEIGEN__::Matrix2x2d PDmPxi = __GEIGEN__::__vec4_to_Mat2x2_double(PDmPx.m[i]);

        __GEIGEN__::Matrix3x2d DsPDminvPxi = __GEIGEN__::__M3x2_M2x2_Multiply(Ds, __GEIGEN__::__M2x2_Mat2x2_multiply(__GEIGEN__::__M2x2_Mat2x2_multiply(DmInv, PDmPxi), DmInv));

        __GEIGEN__::Vector6 tmp = __GEIGEN__::__Mat3x2_to_vec6_double(DsPDminvPxi);
        for (int j = 0;j < 6;j++) {
            DsPDminvPx.m[j][i] = -tmp.v[j];
        }

    }

    return DsPDminvPx;
}

__device__
__GEIGEN__::Matrix9x12d __computePFPX3D_double(const __GEIGEN__::Matrix3x3d& InverseDm) {
    __GEIGEN__::Matrix9x12d matOut;
    __GEIGEN__::__init_Mat9x12_val(matOut, 0);
    double m = InverseDm.m[0][0], n = InverseDm.m[0][1], o = InverseDm.m[0][2];
    double p = InverseDm.m[1][0], q = InverseDm.m[1][1], r = InverseDm.m[1][2];
    double s = InverseDm.m[2][0], t = InverseDm.m[2][1], u = InverseDm.m[2][2];
    double t1 = -(m + p + s);
    double t2 = -(n + q + t);
    double t3 = -(o + r + u);
    matOut.m[0][0] = t1;  matOut.m[0][3] = m;  matOut.m[0][6] = p;  matOut.m[0][9] = s;
    matOut.m[1][1] = t1;  matOut.m[1][4] = m;  matOut.m[1][7] = p;  matOut.m[1][10] = s;
    matOut.m[2][2] = t1;  matOut.m[2][5] = m;  matOut.m[2][8] = p;  matOut.m[2][11] = s;
    matOut.m[3][0] = t2;  matOut.m[3][3] = n;  matOut.m[3][6] = q;  matOut.m[3][9] = t;
    matOut.m[4][1] = t2;  matOut.m[4][4] = n;  matOut.m[4][7] = q;  matOut.m[4][10] = t;
    matOut.m[5][2] = t2;  matOut.m[5][5] = n;  matOut.m[5][8] = q;  matOut.m[5][11] = t;
    matOut.m[6][0] = t3;  matOut.m[6][3] = o;  matOut.m[6][6] = r;  matOut.m[6][9] = u;
    matOut.m[7][1] = t3;  matOut.m[7][4] = o;  matOut.m[7][7] = r;  matOut.m[7][10] = u;
    matOut.m[8][2] = t3;  matOut.m[8][5] = o;  matOut.m[8][8] = r;  matOut.m[8][11] = u;
    return matOut;
}




__device__
void __project_StabbleNHK_H_3D_makePD(__GEIGEN__::Matrix9x9d& H, const __GEIGEN__::Matrix3x3d& F, const __GEIGEN__::Matrix3x3d& sigma, const __GEIGEN__::Matrix3x3d& U, const __GEIGEN__::Matrix3x3d& V, const double& lengthRate, const double& volumRate) {

    double I3 = sigma.m[0][0] * sigma.m[1][1] * sigma.m[2][2];
    double Ic = sigma.m[0][0] * sigma.m[0][0] + sigma.m[1][1] * sigma.m[1][1] + sigma.m[2][2] * sigma.m[2][2];

    double u = lengthRate, r = volumRate;

    __GEIGEN__::Matrix9x9d H1, HJ;//, M9_temp[2];
    __GEIGEN__::__identify_Mat9x9(H1);
    H1 = __GEIGEN__::__S_Mat9x9_multiply(H1, 2);
    __GEIGEN__::Vector9 g = __GEIGEN__::__Mat3x3_to_vec9_double(F);

    __GEIGEN__::Vector9 gJ;
    double3 gjc0 = __GEIGEN__::__v_vec_cross(make_double3(g.v[3], g.v[4], g.v[5]), make_double3(g.v[6], g.v[7], g.v[8]));
    double3 gjc1 = __GEIGEN__::__v_vec_cross(make_double3(g.v[6], g.v[7], g.v[8]), make_double3(g.v[0], g.v[1], g.v[2]));
    double3 gjc2 = __GEIGEN__::__v_vec_cross(make_double3(g.v[0], g.v[1], g.v[2]), make_double3(g.v[3], g.v[4], g.v[5]));
    g = __GEIGEN__::__s_vec9_multiply(g, 2);
    gJ.v[0] = gjc0.x;gJ.v[1] = gjc0.y;gJ.v[2] = gjc0.z;
    gJ.v[3] = gjc1.x;gJ.v[4] = gjc1.y;gJ.v[5] = gjc1.z;
    gJ.v[6] = gjc2.x;gJ.v[7] = gjc2.y;gJ.v[8] = gjc2.z;

    __GEIGEN__::Matrix3x3d f0hat;
    __GEIGEN__::__set_Mat_val(f0hat, 0, -F.m[2][0], F.m[1][0], F.m[2][0], 0, -F.m[0][0], -F.m[1][0], F.m[0][0], 0);

    __GEIGEN__::Matrix3x3d f1hat;
    __GEIGEN__::__set_Mat_val(f1hat, 0, -F.m[2][1], F.m[1][1], F.m[2][1], 0, -F.m[0][1], -F.m[1][1], F.m[0][1], 0);

    __GEIGEN__::Matrix3x3d f2hat;
    __GEIGEN__::__set_Mat_val(f2hat, 0, -F.m[2][2], F.m[1][2], F.m[2][2], 0, -F.m[0][2], -F.m[1][2], F.m[0][2], 0);

    HJ.m[0][0] = 0;HJ.m[0][1] = 0;HJ.m[0][2] = 0;
    HJ.m[1][0] = 0;HJ.m[1][1] = 0;HJ.m[1][2] = 0;
    HJ.m[2][0] = 0;HJ.m[2][1] = 0;HJ.m[2][2] = 0;

    HJ.m[0][3] = -f2hat.m[0][0];HJ.m[0][4] = -f2hat.m[0][1];HJ.m[0][5] = -f2hat.m[0][2];
    HJ.m[1][3] = -f2hat.m[1][0];HJ.m[1][4] = -f2hat.m[1][1];HJ.m[1][5] = -f2hat.m[1][2];
    HJ.m[2][3] = -f2hat.m[2][0];HJ.m[2][4] = -f2hat.m[2][1];HJ.m[2][5] = -f2hat.m[2][2];

    HJ.m[0][6] = f1hat.m[0][0];HJ.m[0][7] = f1hat.m[0][1];HJ.m[0][8] = f1hat.m[0][2];
    HJ.m[1][6] = f1hat.m[1][0];HJ.m[1][7] = f1hat.m[1][1];HJ.m[1][8] = f1hat.m[1][2];
    HJ.m[2][6] = f1hat.m[2][0];HJ.m[2][7] = f1hat.m[2][1];HJ.m[2][8] = f1hat.m[2][2];

    HJ.m[3][0] = f2hat.m[0][0];HJ.m[3][1] = f2hat.m[0][1];HJ.m[3][2] = f2hat.m[0][2];
    HJ.m[4][0] = f2hat.m[1][0];HJ.m[4][1] = f2hat.m[1][1];HJ.m[4][2] = f2hat.m[1][2];
    HJ.m[5][0] = f2hat.m[2][0];HJ.m[5][1] = f2hat.m[2][1];HJ.m[5][2] = f2hat.m[2][2];

    HJ.m[3][3] = 0;HJ.m[3][4] = 0;HJ.m[3][5] = 0;
    HJ.m[4][3] = 0;HJ.m[4][4] = 0;HJ.m[4][5] = 0;
    HJ.m[5][3] = 0;HJ.m[5][4] = 0;HJ.m[5][5] = 0;

    HJ.m[3][6] = -f0hat.m[0][0];HJ.m[3][7] = -f0hat.m[0][1];HJ.m[3][8] = -f0hat.m[0][2];
    HJ.m[4][6] = -f0hat.m[1][0];HJ.m[4][7] = -f0hat.m[1][1];HJ.m[4][8] = -f0hat.m[1][2];
    HJ.m[5][6] = -f0hat.m[2][0];HJ.m[5][7] = -f0hat.m[2][1];HJ.m[5][8] = -f0hat.m[2][2];

    HJ.m[6][0] = -f1hat.m[0][0];HJ.m[6][1] = -f1hat.m[0][1];HJ.m[6][2] = -f1hat.m[0][2];
    HJ.m[7][0] = -f1hat.m[1][0];HJ.m[7][1] = -f1hat.m[1][1];HJ.m[7][2] = -f1hat.m[1][2];
    HJ.m[8][0] = -f1hat.m[2][0];HJ.m[8][1] = -f1hat.m[2][1];HJ.m[8][2] = -f1hat.m[2][2];

    HJ.m[6][3] = f0hat.m[0][0];HJ.m[6][4] = f0hat.m[0][1];HJ.m[6][5] = f0hat.m[0][2];
    HJ.m[7][3] = f0hat.m[1][0];HJ.m[7][4] = f0hat.m[1][1];HJ.m[7][5] = f0hat.m[1][2];
    HJ.m[8][3] = f0hat.m[2][0];HJ.m[8][4] = f0hat.m[2][1];HJ.m[8][5] = f0hat.m[2][2];

    HJ.m[6][6] = 0;HJ.m[6][7] = 0;HJ.m[6][8] = 0;
    HJ.m[7][6] = 0;HJ.m[7][7] = 0;HJ.m[7][8] = 0;
    HJ.m[8][6] = 0;HJ.m[8][7] = 0;HJ.m[8][8] = 0;

    double J = I3;
    double mu = u, lambda = r;
    H = __GEIGEN__::__Mat9x9_add(__GEIGEN__::__S_Mat9x9_multiply(H1, (Ic * mu) / (2 * (Ic + 1))), __GEIGEN__::__S_Mat9x9_multiply(HJ, lambda * (J - 1 - (3 * mu) / (4.0 * lambda))));

    H = __GEIGEN__::__Mat9x9_add(H, __GEIGEN__::__v9_vec9_toMat9x9(g, g, (mu / (2 * (Ic + 1) * (Ic + 1)))));

    H = __GEIGEN__::__Mat9x9_add(H, __GEIGEN__::__v9_vec9_toMat9x9(gJ, gJ,lambda));

    Eigen::Matrix<double, 9, 9> mat9;
    for (int i = 0; i != 9; ++i)
        for (int j = 0; j != 9; ++j)
            mat9(i, j) = H.m[i][j];

    PDSNK<double, 9>(mat9);

    for (int i = 0; i != 9; ++i)
        for (int j = 0; j != 9; ++j)
            H.m[i][j] = mat9(i, j);

}

__global__
void _calculate_fem_gradient_hessian(__GEIGEN__::Matrix3x3d* DmInverses, const double3* vertexes, const uint4* tetrahedras,
    __GEIGEN__::Matrix12x12d* Hessians, uint32_t offset, const double* volume, double3* gradient, int tetrahedraNum, double lenRate, double volRate, double IPC_dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tetrahedraNum) return;

    __GEIGEN__::Matrix9x12d PFPX = __computePFPX3D_double(DmInverses[idx]);

    __GEIGEN__::Matrix3x3d Ds;
    __calculateDms3D_double(vertexes, tetrahedras[idx], Ds);
    __GEIGEN__::Matrix3x3d F;
    __M_Mat_multiply(Ds, DmInverses[idx], F);

    __GEIGEN__::Matrix3x3d U, V, Sigma;
    SVD(F, U, V, Sigma);

#ifdef USE_SNK
    __GEIGEN__::Matrix3x3d Iso_PEPF = __computePEPF_StableNHK3D_double(F, Sigma, U, V, lenRate, volRate);
#else
    __GEIGEN__::Matrix3x3d Iso_PEPF = computePEPF_ARAP_double(F, Sigma, U, V, lenRate);
#endif


    __GEIGEN__::Matrix3x3d PEPF = Iso_PEPF;

    __GEIGEN__::Vector9 pepf = __GEIGEN__::__Mat3x3_to_vec9_double(PEPF);




    __GEIGEN__::Matrix12x9d PFPXTranspose = __GEIGEN__::__Transpose9x12(PFPX);
    __GEIGEN__::Vector12 f = __GEIGEN__::__s_vec12_multiply(__GEIGEN__::__M12x9_v9_multiply(PFPXTranspose, pepf), IPC_dt * IPC_dt * volume[idx]);
    //printf("%f  %f  %f  %f  %f  %f  %f  %f  %f  %f  %f  %f\n", f.v[0], f.v[1], f.v[2], f.v[3], f.v[4], f.v[5], f.v[6], f.v[7], f.v[8], f.v[9], f.v[10], f.v[11]);

    {
        gipc::ATOMIC_ADD(&(gradient[tetrahedras[idx].x].x), f.v[0]);
        gipc::ATOMIC_ADD(&(gradient[tetrahedras[idx].x].y), f.v[1]);
        gipc::ATOMIC_ADD(&(gradient[tetrahedras[idx].x].z), f.v[2]);

        gipc::ATOMIC_ADD(&(gradient[tetrahedras[idx].y].x), f.v[3]);
        gipc::ATOMIC_ADD(&(gradient[tetrahedras[idx].y].y), f.v[4]);
        gipc::ATOMIC_ADD(&(gradient[tetrahedras[idx].y].z), f.v[5]);

        gipc::ATOMIC_ADD(&(gradient[tetrahedras[idx].z].x), f.v[6]);
        gipc::ATOMIC_ADD(&(gradient[tetrahedras[idx].z].y), f.v[7]);
        gipc::ATOMIC_ADD(&(gradient[tetrahedras[idx].z].z), f.v[8]);

        gipc::ATOMIC_ADD(&(gradient[tetrahedras[idx].w].x), f.v[9]);
        gipc::ATOMIC_ADD(&(gradient[tetrahedras[idx].w].y), f.v[10]);
        gipc::ATOMIC_ADD(&(gradient[tetrahedras[idx].w].z), f.v[11]);
    }

#ifdef USE_SNK
    __GEIGEN__::Matrix9x9d Hq;
    //__project_StabbleNHK_H_3D(make_double3(Sigma.m[0][0], Sigma.m[1][1], Sigma.m[2][2]), U, V, lenRate, volRate,Hq);

    __project_StabbleNHK_H_3D_makePD(Hq, F, Sigma, U, V, lenRate, volRate);
#else
    __GEIGEN__::Matrix9x9d Hq = project_ARAP_H_3D(Sigma, U, V, lenRate);
#endif


    __GEIGEN__::Matrix12x12d H;

    __GEIGEN__::__M12x9_S9x9_MT9x12_Multiply(PFPXTranspose, Hq, H);

    Hessians[idx + offset] = __GEIGEN__::__s_M12x12_Multiply(H, volume[idx] * IPC_dt * IPC_dt);
}


__global__
void _calculate_triangle_fem_gradient_hessian(__GEIGEN__::Matrix2x2d* trimInverses, const double3* vertexes, const uint3* triangles,
    __GEIGEN__::Matrix9x9d* Hessians, uint32_t offset, const double* area, double3* gradient, int triangleNum, double stretchStiff, double shearhStiff, double IPC_dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= triangleNum) return;

    __GEIGEN__::Matrix6x9d PFPX = __computePFDsPX3D_6x9_double(trimInverses[idx]);

    __GEIGEN__::Matrix3x2d Ds;
    __calculateDs2D_double(vertexes, triangles[idx], Ds);
    __GEIGEN__::Matrix3x2d F = __GEIGEN__::__M3x2_M2x2_Multiply(Ds, trimInverses[idx]);

    __GEIGEN__::Matrix3x2d PEPF = __computePEPF_BaraffWitkinStretch_double(F, stretchStiff, shearhStiff);



    __GEIGEN__::Vector6 pepf = __GEIGEN__::__Mat3x2_to_vec6_double(PEPF);




    __GEIGEN__::Matrix9x6d PFPXTranspose = __GEIGEN__::__Transpose6x9(PFPX);
    __GEIGEN__::Vector9 f = __GEIGEN__::__s_vec9_multiply(__GEIGEN__::__M9x6_v6_multiply(PFPXTranspose, pepf), IPC_dt * IPC_dt * area[idx]);
    //printf("%f  %f  %f  %f  %f  %f  %f  %f  %f  %f  %f  %f\n", f.v[0], f.v[1], f.v[2], f.v[3], f.v[4], f.v[5], f.v[6], f.v[7], f.v[8], f.v[9], f.v[10], f.v[11]);

    {
        gipc::ATOMIC_ADD(&(gradient[triangles[idx].x].x), f.v[0]);
        gipc::ATOMIC_ADD(&(gradient[triangles[idx].x].y), f.v[1]);
        gipc::ATOMIC_ADD(&(gradient[triangles[idx].x].z), f.v[2]);
        gipc::ATOMIC_ADD(&(gradient[triangles[idx].y].x), f.v[3]);
        gipc::ATOMIC_ADD(&(gradient[triangles[idx].y].y), f.v[4]);
        gipc::ATOMIC_ADD(&(gradient[triangles[idx].y].z), f.v[5]);
        gipc::ATOMIC_ADD(&(gradient[triangles[idx].z].x), f.v[6]);
        gipc::ATOMIC_ADD(&(gradient[triangles[idx].z].y), f.v[7]);
        gipc::ATOMIC_ADD(&(gradient[triangles[idx].z].z), f.v[8]);
    }

    __GEIGEN__::Matrix6x6d Hq = __GEIGEN__::__s_M6x6_Multiply(__project_BaraffWitkinStretch_H(F), stretchStiff);

    __GEIGEN__::__Mat_add(Hq, __GEIGEN__::__s_M6x6_Multiply(__project_BaraffWitkinShear_H(F), shearhStiff), Hq);

    __GEIGEN__::Matrix9x6d M9x6_temp = __GEIGEN__::__M9x6_M6x6_Multiply(PFPXTranspose, Hq);
    __GEIGEN__::Matrix9x9d H = __GEIGEN__::__M9x6_M6x9_Multiply(M9x6_temp, PFPX);
    H = __GEIGEN__::__s_M9x9_Multiply(H, area[idx] * IPC_dt * IPC_dt);
    Hessians[idx + offset] = H;
}

__global__
void _calculate_triangle_fem_gradient(__GEIGEN__::Matrix2x2d* trimInverses, const double3* vertexes, const uint3* triangles,
    const double* area, double3* gradient, int triangleNum, double stretchStiff, double shearhStiff, double IPC_dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= triangleNum) return;

    __GEIGEN__::Matrix6x9d PFPX = __computePFDsPX3D_6x9_double(trimInverses[idx]);

    __GEIGEN__::Matrix3x2d Ds;
    __calculateDs2D_double(vertexes, triangles[idx], Ds);
    __GEIGEN__::Matrix3x2d F = __GEIGEN__::__M3x2_M2x2_Multiply(Ds, trimInverses[idx]);

    __GEIGEN__::Matrix3x2d PEPF = __computePEPF_BaraffWitkinStretch_double(F, stretchStiff, shearhStiff);



    __GEIGEN__::Vector6 pepf = __GEIGEN__::__Mat3x2_to_vec6_double(PEPF);




    __GEIGEN__::Matrix9x6d PFPXTranspose = __GEIGEN__::__Transpose6x9(PFPX);
    __GEIGEN__::Vector9 f = __GEIGEN__::__s_vec9_multiply(__GEIGEN__::__M9x6_v6_multiply(PFPXTranspose, pepf), IPC_dt * IPC_dt * area[idx]);
    //printf("%f  %f  %f  %f  %f  %f  %f  %f  %f  %f  %f  %f\n", f.v[0], f.v[1], f.v[2], f.v[3], f.v[4], f.v[5], f.v[6], f.v[7], f.v[8], f.v[9], f.v[10], f.v[11]);

    {
        gipc::ATOMIC_ADD(&(gradient[triangles[idx].x].x), f.v[0]);
        gipc::ATOMIC_ADD(&(gradient[triangles[idx].x].y), f.v[1]);
        gipc::ATOMIC_ADD(&(gradient[triangles[idx].x].z), f.v[2]);
        gipc::ATOMIC_ADD(&(gradient[triangles[idx].y].x), f.v[3]);
        gipc::ATOMIC_ADD(&(gradient[triangles[idx].y].y), f.v[4]);
        gipc::ATOMIC_ADD(&(gradient[triangles[idx].y].z), f.v[5]);
        gipc::ATOMIC_ADD(&(gradient[triangles[idx].z].x), f.v[6]);
        gipc::ATOMIC_ADD(&(gradient[triangles[idx].z].y), f.v[7]);
        gipc::ATOMIC_ADD(&(gradient[triangles[idx].z].z), f.v[8]);
    }
}

__global__
void _calculate_fem_gradient(__GEIGEN__::Matrix3x3d* DmInverses, const double3* vertexes, const uint4* tetrahedras,
    const double* volume, double3* gradient, int tetrahedraNum, double lenRate, double volRate, double dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tetrahedraNum) return;

    __GEIGEN__::Matrix9x12d PFPX = __computePFPX3D_double(DmInverses[idx]);

    __GEIGEN__::Matrix3x3d Ds;
    __calculateDms3D_double(vertexes, tetrahedras[idx], Ds);
    __GEIGEN__::Matrix3x3d F;
    __M_Mat_multiply(Ds, DmInverses[idx], F);

    __GEIGEN__::Matrix3x3d U, V, Sigma;
    SVD(F, U, V, Sigma);
    //printf("%f %f\n\n\n", lenRate, volRate);
#ifdef USE_SNK
    __GEIGEN__::Matrix3x3d Iso_PEPF = __computePEPF_StableNHK3D_double(F, Sigma, U, V, lenRate, volRate);
#else
    __GEIGEN__::Matrix3x3d Iso_PEPF = computePEPF_ARAP_double(F, Sigma, U, V, lenRate);
#endif



    __GEIGEN__::Matrix3x3d PEPF = Iso_PEPF;

    __GEIGEN__::Vector9 pepf = __GEIGEN__::__Mat3x3_to_vec9_double(PEPF);

    __GEIGEN__::Matrix12x9d PFPXTranspose = __GEIGEN__::__Transpose9x12(PFPX);

    __GEIGEN__::Vector12 f = __GEIGEN__::__M12x9_v9_multiply(PFPXTranspose, pepf);

    for (int i = 0; i < 12; i++) {
        f.v[i] = volume[idx] * f.v[i] * dt * dt;
    }

    {
        gipc::ATOMIC_ADD(&(gradient[tetrahedras[idx].x].x), f.v[0]);
        gipc::ATOMIC_ADD(&(gradient[tetrahedras[idx].x].y), f.v[1]);
        gipc::ATOMIC_ADD(&(gradient[tetrahedras[idx].x].z), f.v[2]);

        gipc::ATOMIC_ADD(&(gradient[tetrahedras[idx].y].x), f.v[3]);
        gipc::ATOMIC_ADD(&(gradient[tetrahedras[idx].y].y), f.v[4]);
        gipc::ATOMIC_ADD(&(gradient[tetrahedras[idx].y].z), f.v[5]);

        gipc::ATOMIC_ADD(&(gradient[tetrahedras[idx].z].x), f.v[6]);
        gipc::ATOMIC_ADD(&(gradient[tetrahedras[idx].z].y), f.v[7]);
        gipc::ATOMIC_ADD(&(gradient[tetrahedras[idx].z].z), f.v[8]);

        gipc::ATOMIC_ADD(&(gradient[tetrahedras[idx].w].x), f.v[9]);
        gipc::ATOMIC_ADD(&(gradient[tetrahedras[idx].w].y), f.v[10]);
        gipc::ATOMIC_ADD(&(gradient[tetrahedras[idx].w].z), f.v[11]);
    }
}





using namespace Eigen;
__global__
void _calculate_bending_gradient_hessian(const double3* vertexes, const double3* rest_vertexes, const uint2* edges, const uint2* edges_adj_vertex,
    __GEIGEN__::Matrix12x12d* Hessians, uint4* Indices, uint32_t offset, double3* gradient, int edgeNum, double bendStiff, double IPC_dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= edgeNum) return;
    uint2 edge = edges[idx];
    uint2 adj = edges_adj_vertex[idx];
    if (adj.y == -1) {
        __GEIGEN__::Matrix12x12d Zero;
        for (int i = 0; i < 12; i++) {
            for (int j = 0; j < 12; j++) {
                Zero.m[i][j] = 0;
            }
        }
        Hessians[idx + offset] = Zero;
        Indices[idx + offset] = make_uint4(0, 1, 2, 3);
        //
        return;
    }
    auto x0 = vertexes[edge.x];
    auto x1 = vertexes[edge.y];
    auto x2 = vertexes[adj.x];
    auto x3 = vertexes[adj.y];

    Matrix<double, 1, 12> grad_transpose;
    Matrix<double, 12, 12> H;

    Eigen::Matrix<double, 3, 1> x0_eigen = Eigen::Matrix<double, 3, 1>(x0.x, x0.y, x0.z);
    Eigen::Matrix<double, 3, 1> x1_eigen = Eigen::Matrix<double, 3, 1>(x1.x, x1.y, x1.z);
    Eigen::Matrix<double, 3, 1> x2_eigen = Eigen::Matrix<double, 3, 1>(x2.x, x2.y, x2.z);
    Eigen::Matrix<double, 3, 1> x3_eigen = Eigen::Matrix<double, 3, 1>(x3.x, x3.y, x3.z);
    double t = LibShell::edgeTheta(x0_eigen, x1_eigen, x2_eigen, x3_eigen, &grad_transpose, &H);
    //            cout << "t: " << t << endl;

    auto rest_x0 = rest_vertexes[edge.x];
    auto rest_x1 = rest_vertexes[edge.y];
    auto rest_x2 = rest_vertexes[adj.x];
    auto rest_x3 = rest_vertexes[adj.y];
    double length = __GEIGEN__::__norm(__GEIGEN__::__minus(rest_x0, rest_x1));
    Eigen::Vector3d rest_x0_eigen = Eigen::Vector3d(rest_x0.x, rest_x0.y, rest_x0.z);
    Eigen::Vector3d rest_x1_eigen = Eigen::Vector3d(rest_x1.x, rest_x1.y, rest_x1.z);
    Eigen::Vector3d rest_x2_eigen = Eigen::Vector3d(rest_x2.x, rest_x2.y, rest_x2.z);
    Eigen::Vector3d rest_x3_eigen = Eigen::Vector3d(rest_x3.x, rest_x3.y, rest_x3.z);
    double rest_t = LibShell::edgeTheta(rest_x0_eigen, rest_x1_eigen, rest_x2_eigen, rest_x3_eigen, nullptr, nullptr);

    H = 2 * ((t - rest_t) * H + grad_transpose.transpose() * grad_transpose);
    grad_transpose = 2 * (t - rest_t) * grad_transpose;
    PDSNK<double, 12>(H);
    __GEIGEN__::Vector12 f;
    for (int i = 0;i < 12;i++) {
        f.v[i] = IPC_dt * IPC_dt * length * grad_transpose(0, i) * bendStiff;
    }


    {
        gipc::ATOMIC_ADD(&(gradient[edge.x].x), f.v[0]);
        gipc::ATOMIC_ADD(&(gradient[edge.x].y), f.v[1]);
        gipc::ATOMIC_ADD(&(gradient[edge.x].z), f.v[2]);

        gipc::ATOMIC_ADD(&(gradient[edge.y].x), f.v[3]);
        gipc::ATOMIC_ADD(&(gradient[edge.y].y), f.v[4]);
        gipc::ATOMIC_ADD(&(gradient[edge.y].z), f.v[5]);

        gipc::ATOMIC_ADD(&(gradient[adj.x].x), f.v[6]);
        gipc::ATOMIC_ADD(&(gradient[adj.x].y), f.v[7]);
        gipc::ATOMIC_ADD(&(gradient[adj.x].z), f.v[8]);

        gipc::ATOMIC_ADD(&(gradient[adj.y].x), f.v[9]);
        gipc::ATOMIC_ADD(&(gradient[adj.y].y), f.v[10]);
        gipc::ATOMIC_ADD(&(gradient[adj.y].z), f.v[11]);
    }
    __GEIGEN__::Matrix12x12d d_H;
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 12; j++) {
            d_H.m[i][j] = IPC_dt * IPC_dt * length * H(i, j) * bendStiff;
        }
    }
    Hessians[idx + offset] = d_H;
    Indices[idx + offset] = make_uint4(edge.x, edge.y, adj.x, adj.y);
}
__device__
double __cal_bending_energy(const double3* vertexes, const double3* rest_vertexes, const uint2& edge, const uint2& adj, double length, double bendStiff)
{
    if (adj.y == -1)return 0;
    auto x0 = vertexes[edge.x];
    auto x1 = vertexes[edge.y];
    auto x2 = vertexes[adj.x];
    auto x3 = vertexes[adj.y];
    Eigen::Matrix<double, 3, 1> x0_eigen = Eigen::Matrix<double, 3, 1>(x0.x, x0.y, x0.z);
    Eigen::Matrix<double, 3, 1> x1_eigen = Eigen::Matrix<double, 3, 1>(x1.x, x1.y, x1.z);
    Eigen::Matrix<double, 3, 1> x2_eigen = Eigen::Matrix<double, 3, 1>(x2.x, x2.y, x2.z);
    Eigen::Matrix<double, 3, 1> x3_eigen = Eigen::Matrix<double, 3, 1>(x3.x, x3.y, x3.z);
    double t = LibShell::edgeTheta(x0_eigen, x1_eigen, x2_eigen, x3_eigen, nullptr, nullptr);
    //            cout << "t: " << t << endl;

    auto rest_x0 = rest_vertexes[edge.x];
    auto rest_x1 = rest_vertexes[edge.y];
    auto rest_x2 = rest_vertexes[adj.x];
    auto rest_x3 = rest_vertexes[adj.y];
    Eigen::Matrix<double, 3, 1> rest_x0_eigen = Eigen::Matrix<double, 3, 1>(rest_x0.x, rest_x0.y, rest_x0.z);
    Eigen::Matrix<double, 3, 1> rest_x1_eigen = Eigen::Matrix<double, 3, 1>(rest_x1.x, rest_x1.y, rest_x1.z);
    Eigen::Matrix<double, 3, 1> rest_x2_eigen = Eigen::Matrix<double, 3, 1>(rest_x2.x, rest_x2.y, rest_x2.z);
    Eigen::Matrix<double, 3, 1> rest_x3_eigen = Eigen::Matrix<double, 3, 1>(rest_x3.x, rest_x3.y, rest_x3.z);
    double rest_t = LibShell::edgeTheta(rest_x0_eigen, rest_x1_eigen, rest_x2_eigen, rest_x3_eigen, nullptr, nullptr);
    double bend_energy = bendStiff * (t - rest_t) * (t - rest_t) * length;
    return bend_energy;
}











double calculateVolum(const double3* vertexes, const uint4& index) {
    int id0 = 0;
    int id1 = 1;
    int id2 = 2;
    int id3 = 3;
    double o1x = vertexes[index.y].x - vertexes[index.x].x;
    double o1y = vertexes[index.y].y - vertexes[index.x].y;
    double o1z = vertexes[index.y].z - vertexes[index.x].z;
    double3 OA = make_double3(o1x, o1y, o1z);

    double o2x = vertexes[index.z].x - vertexes[index.x].x;
    double o2y = vertexes[index.z].y - vertexes[index.x].y;
    double o2z = vertexes[index.z].z - vertexes[index.x].z;
    double3 OB = make_double3(o2x, o2y, o2z);

    double o3x = vertexes[index.w].x - vertexes[index.x].x;
    double o3y = vertexes[index.w].y - vertexes[index.x].y;
    double o3z = vertexes[index.w].z - vertexes[index.x].z;
    double3 OC = make_double3(o3x, o3y, o3z);

    double3 heightDir = __GEIGEN__::__v_vec_cross(OA, OB);//OA.cross(OB);
    double bottomArea = __GEIGEN__::__norm(heightDir);//heightDir.norm();
    heightDir = __GEIGEN__::__normalized(heightDir);

    double volum = bottomArea * __GEIGEN__::__v_vec_dot(heightDir, OC) / 6;
    return volum > 0 ? volum : -volum;
}

double calculateArea(const double3* vertexes, const uint3& index)
{
    //double a = __GEIGEN__::__norm(__GEIGEN__::__minus(vertexes[index.y], vertexes[index.x]));
    //double b = __GEIGEN__::__norm(__GEIGEN__::__minus(vertexes[index.z], vertexes[index.x]));
    //double c = __GEIGEN__::__norm(__GEIGEN__::__minus(vertexes[index.y], vertexes[index.z]));
    //double s = (a + b + c) / 2;
    //return sqrt(s * (s - a) *
    //    (s - b) * (s - c));
    double3 v10 = __GEIGEN__::__minus(vertexes[index.y], vertexes[index.x]);
    double3 v20 = __GEIGEN__::__minus(vertexes[index.z], vertexes[index.x]);
    double area = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v10, v20));
    return 0.5 * area;
}
