//
// gl_main.cpp
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include "GL/glew.h"
#include "GL/freeglut.h"
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
#include <map>
#include "GIPC.cuh"
#include "device_launch_parameters.h"
#include "mlbvh.cuh"
#include <stdio.h>
#include "load_mesh.h"
#include "cuda_tools.h"
#include <queue>
//#include "timer.h"
#include "femEnergy.cuh"
#include "gpu_eigen_libs.cuh"
#include "fem_parameters.h"
#include "gipc_path.h"


int collision_detection_buff_scale = 1;
double motion_rate = 1;
mesh_obj obj;
lbvh_f bvh_f;
lbvh_e bvh_e;
GIPC ipc;
device_TetraData d_tetMesh;
tetrahedra_obj tetMesh;
vector<Node> nodes;
vector<AABB> bvs;
vector<string> obj_pathes;
int initPath = 0;
using namespace std;
int step = 0;
int frameId = 0;
int surfNumId = 0;
float xRot = 0.0f;
float yRot = 0.f;
float xTrans = 0;
float yTrans = 0;
float zTrans = 0;
int ox;
int oy;
int buttonState;
float xRotLength = 0.0f;
float yRotLength = 0.0f;
float window_width = 1000;
float window_height = 1000;
int s_dimention = 3;
bool saveSurface = false;
bool change = false;
bool screenshot = false;

bool drawbvh = false;
bool drawSurface = true;

bool stop = true;

double3 center;
double3 Ssize;

GLuint PN_vbo_;
GLuint VAO;
GLuint color_vbo_;
//GLuint color_vao_;
GLuint normal_vbo_;
//GLuint normal_vao_;
GLuint v;
GLuint f;
GLuint shaderProgram;

int clothFaceOffset = 0;
int bodyVertOffset = 0;



void Init_CUDA() {
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		exit(0);
	}
}

//typedef struct BITMAPFILEHEADER
//{
//    u_int16_t bfType;
//    u_int32_t bfSize;
//    u_int16_t bfReserved1;
//    u_int16_t bfReserved2;
//    u_int32_t bfOffBits;
//}BITMAPFILEHEADER;
//
//typedef struct BITMAPINFOHEADER
//{
//    u_int32_t biSize;
//    u_int32_t biWidth;
//    u_int32_t biHeight;
//    u_int16_t biPlanes;
//    u_int16_t biBitCount;
//    u_int32_t biCompression;
//    u_int32_t biSizeImage;
//    u_int32_t biXPelsPerMeter;
//    u_int32_t biYPelsPerMeter;
//    u_int32_t biClrUsed;
//    u_int32_t biClrImportant;
//}BITMAPINFODEADER;

bool WriteBitmapFile(int width, int height, const std::string& file_name, unsigned char* bitmapData)
{
#ifdef WIN32
	BITMAPFILEHEADER bitmapFileHeader;
	memset(&bitmapFileHeader, 0, sizeof(BITMAPFILEHEADER));
	bitmapFileHeader.bfSize = sizeof(BITMAPFILEHEADER);
	bitmapFileHeader.bfType = 0x4d42;   //BM
	bitmapFileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);

	BITMAPINFOHEADER bitmapInfoHeader;
	memset(&bitmapInfoHeader, 0, sizeof(BITMAPINFOHEADER));
	bitmapInfoHeader.biSize = sizeof(BITMAPINFOHEADER);
	bitmapInfoHeader.biWidth = width;
	bitmapInfoHeader.biHeight = height;
	bitmapInfoHeader.biPlanes = 1;
	bitmapInfoHeader.biBitCount = 24;
	bitmapInfoHeader.biCompression = 0L;
	bitmapInfoHeader.biSizeImage = width * abs(height) * 3;

	//////////////////////////////////////////////////////////////////////////
	FILE* filePtr;
	unsigned char tempRGB;
	int imageIdx;

	for (imageIdx = 0; imageIdx < (int)bitmapInfoHeader.biSizeImage; imageIdx += 3)
	{
		tempRGB = bitmapData[imageIdx];
		bitmapData[imageIdx] = bitmapData[imageIdx + 2];
		bitmapData[imageIdx + 2] = tempRGB;
	}

	filePtr = fopen(file_name.c_str(), "wb");
	if (NULL == filePtr)
	{
		return false;
	}

	fwrite(&bitmapFileHeader, sizeof(BITMAPFILEHEADER), 1, filePtr);

	fwrite(&bitmapInfoHeader, sizeof(BITMAPINFOHEADER), 1, filePtr);

	fwrite(bitmapData, bitmapInfoHeader.biSizeImage, 1, filePtr);

	fclose(filePtr);
#endif
	return true;
}

void SaveScreenShot(int width, int height, const std::string& file_name)
{
#ifdef WIN32
	int data_len = height * width * 3;      // bytes
	void* screen_data = malloc(data_len);
	memset(screen_data, 0, data_len);
	glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, screen_data);

	WriteBitmapFile(width, height, file_name + ".bmp", (unsigned char*)screen_data);

	free(screen_data);
#endif
}

void saveSurfaceMesh(const string& path) {
	std::stringstream ss;
	ss << path;
	ss.fill('0');
	ss.width(5);
	ss << (surfNumId++) / 1;// / 10;
	//if (surfNumId % 10 != 0) return;
	ss << ".obj";
	std::string file_path = ss.str();
	ofstream outSurf(file_path);

	map<int, int> meshToSurf;
	for (int i = 0; i < tetMesh.surfVerts.size(); i++) {
		const auto& pos = tetMesh.vertexes[tetMesh.surfVerts[i]];
		outSurf << "v " << pos.x << " " << pos.y << " " << pos.z << endl;
		meshToSurf[tetMesh.surfVerts[i]] = i;
	}

	for (int i = 0; i < tetMesh.surface.size(); i++) {
		const auto& tri = tetMesh.surface[i];
		outSurf << "f " << meshToSurf[tri.x] + 1 << " " << meshToSurf[tri.y] + 1 << " " << meshToSurf[tri.z] + 1 << endl;
	}
	outSurf.close();
}


void draw_box2D(float ox, float oy, float width, float height)
{
	glLineWidth(2.5f);
	glColor3f(0.8f, 0.8f, 0.8f);

	glBegin(GL_LINES);

	glVertex3f(ox, oy, 0);
	glVertex3f(ox + width, oy, 0);

	glVertex3f(ox, oy, 0);
	glVertex3f(ox, oy + height, 0);

	glVertex3f(ox + width, oy, 0);
	glVertex3f(ox + width, oy + height, 0);

	glVertex3f(ox + width, oy + height, 0);
	glVertex3f(ox, oy + height, 0);

	glEnd();
}

void draw_box3D(float ox, float oy, float oz, float width, float height, float length, int boxType = 0)
{
	glLineWidth(0.5f);
	glColor3f(0.8f, 0.8f, 0.1f);
	if (boxType == 1) {
		glLineWidth(1.5f);
		glColor3f(0.8f, 0.8f, 0.8f);
	}
	glBegin(GL_LINES);

	glVertex3f(ox, oy, oz);
	glVertex3f(ox + width, oy, oz);

	glVertex3f(ox, oy, oz);
	glVertex3f(ox, oy + height, oz);

	glVertex3f(ox, oy, oz);
	glVertex3f(ox, oy, oz + length);

	glVertex3f(ox + width, oy, oz);
	glVertex3f(ox + width, oy + height, oz);

	glVertex3f(ox + width, oy + height, oz);
	glVertex3f(ox, oy + height, oz);

	glVertex3f(ox, oy + height, oz + length);
	glVertex3f(ox, oy, oz + length);

	glVertex3f(ox, oy + height, oz + length);
	glVertex3f(ox, oy + height, oz);

	glVertex3f(ox + width, oy, oz);
	glVertex3f(ox + width, oy, oz + length);

	glVertex3f(ox, oy, oz + length);
	glVertex3f(ox + width, oy, oz + length);

	glVertex3f(ox + width, oy + height, oz);
	glVertex3f(ox + width, oy + height, oz + length);

	glVertex3f(ox + width, oy + height, oz + length);
	glVertex3f(ox + width, oy, oz + length);

	glVertex3f(ox, oy + height, oz + length);
	glVertex3f(ox + width, oy + height, oz + length);

	glEnd();
}

void draw_lines(float ox, float oy, float oz, float width, float height, float length)
{
	glLineWidth(0.5f);
	glColor3f(0.8f, 0.8f, 0.8f);

	glBegin(GL_LINES);
	int numbers = 20;
	for (int i = 0; i <= numbers; i++) {
		//glVertex3f(ox, oy, oz);
		glVertex3f(ox + width * i / numbers, oy, 0);
		glVertex3f(ox + width * i / numbers, oy + height, 0);
	}

	for (int i = 0; i <= numbers; i++) {
		//glVertex3f(ox, oy, oz);
		glVertex3f(ox, oy + height * i / numbers, 0);
		glVertex3f(ox + width, oy + height * i / numbers, 0);
	}

	glEnd();


	glLineWidth(1.5f);
	glColor3f(0.8f, 0.8f, 0.f);
	glBegin(GL_LINES);
	glVertex3f(ox + width / 2, oy, 0);
	glVertex3f(ox + width / 2, oy + height, 0);

	glVertex3f(ox, oy + height / 2, 0);
	glVertex3f(ox + width, oy + height / 2, 0);

	glEnd();
}

void draw_mesh3D()
{
	glEnable(GL_DEPTH_TEST);
	glLineWidth(1.5f);
	glColor3f(0.9f, 0.1f, 0.1f);
	const vector<uint3>& surf = tetMesh.surface;//obj.faces;
	glBegin(GL_TRIANGLES);



	for (int j = 0; j < tetMesh.surface.size(); j++) {
		glVertex3f((tetMesh.vertexes[surf[j].x].x), (tetMesh.vertexes[surf[j].x].y), (tetMesh.vertexes[surf[j].x].z));
		glVertex3f((tetMesh.vertexes[surf[j].y].x), (tetMesh.vertexes[surf[j].y].y), (tetMesh.vertexes[surf[j].y].z));
		glVertex3f((tetMesh.vertexes[surf[j].z].x), (tetMesh.vertexes[surf[j].z].y), (tetMesh.vertexes[surf[j].z].z));
	}
	glEnd();

	glColor3f(0.9f, 0.9f, 0.9f);
	//glDisable(GL_DEPTH_TEST);
	glLineWidth(0.1f);
	glBegin(GL_LINES);

	for (int j = 0; j < tetMesh.surfEdges.size(); j++) {
		//if ((tetMesh.surfEdges[j].x == 870 && tetMesh.surfEdges[j].y == 965) || (tetMesh.surfEdges[j].x == 965 && tetMesh.surfEdges[j].y == 870)) {          
		//    glColor3f(0.9f, 0.1f, 0.1f);
		//    glLineWidth(3.4f);
		//}
		//else if ((tetMesh.surfEdges[j].x == 870 && tetMesh.surfEdges[j].y == 905) || (tetMesh.surfEdges[j].x == 905 && tetMesh.surfEdges[j].y == 870)) {
		//    glColor3f(0.9f, 0.9f, 0.1f);
		//    glLineWidth(3.4f);
		//}

		glVertex3f((tetMesh.vertexes[tetMesh.surfEdges[j].x].x), (tetMesh.vertexes[tetMesh.surfEdges[j].x].y), (tetMesh.vertexes[tetMesh.surfEdges[j].x].z));
		glVertex3f((tetMesh.vertexes[tetMesh.surfEdges[j].y].x), (tetMesh.vertexes[tetMesh.surfEdges[j].y].y), (tetMesh.vertexes[tetMesh.surfEdges[j].y].z));

		glColor3f(0.9f, 0.9f, 0.9f);
		glLineWidth(0.1f);
	}
	glEnd();

	//glColor3f(0.99f, 0.1f, 0.1f);
	////glDisable(GL_DEPTH_TEST);
	//glPointSize(8);
	//glBegin(GL_POINTS);
	//glVertex3f((tetMesh.vertexes[2189].x), (tetMesh.vertexes[2189].y), (tetMesh.vertexes[2189].z));
	//glColor3f(0.99f, 0.99f, 0.1f);
	//glVertex3f((tetMesh.vertexes[870].x), (tetMesh.vertexes[870].y), (tetMesh.vertexes[870].z));
	//glVertex3f((tetMesh.vertexes[905].x), (tetMesh.vertexes[905].y), (tetMesh.vertexes[905].z));
	//glVertex3f((tetMesh.vertexes[965].x), (tetMesh.vertexes[965].y), (tetMesh.vertexes[965].z));
	//glEnd();
}

void draw_bvh() {
	int num = (bvs.size() + 1) / 2;
	for (int j = 0; j < bvs.size(); j++) {
		int i = j;
		float ox, oy, oz, bwidth, bheight, blength;
		ox = (bvs[i].lower.x);
		oy = (bvs[i].lower.y);
		oz = (bvs[i].lower.z);
		bwidth = (bvs[i].upper.x - bvs[i].lower.x);
		bheight = (bvs[i].upper.y - bvs[i].lower.y);
		blength = (bvs[i].upper.z - bvs[i].lower.z);
		draw_box3D(ox, oy, oz, bwidth, bheight, blength);
	}
}

int counttt = 0;
vector<float3> getRenderGeometry(int& number) {

	vector<double3> meshNormal(tetMesh.vertexNum, make_double3(0, 0, 0));
	number = tetMesh.surface.size();//meshTemp.surfaceRender.size();
	vector<float3> pos_normal_color(3 * number * 3);

	for (int i = 0; i < number; i++)
	{
		//int tetId = meshTemp.surfaceRender[i][3];
		int v0 = tetMesh.surface[i].x;
		int v1 = tetMesh.surface[i].y;
		int v2 = tetMesh.surface[i].z;
		double3 vt0 = tetMesh.vertexes[v0];// Vector3d(meshTemp.vertexes[v0][0], meshTemp.vertexes[v0][1], meshTemp.vertexes[v0][2]);
		double3 vt1 = tetMesh.vertexes[v1];// Vector3d(meshTemp.vertexes[v1][0], meshTemp.vertexes[v1][1], meshTemp.vertexes[v1][2]);
		double3 vt2 = tetMesh.vertexes[v2];// Vector3d(meshTemp.vertexes[v2][0], meshTemp.vertexes[v2][1], meshTemp.vertexes[v2][2]);
		double3 vec1 = __GEIGEN__::__minus(vt1, vt0);//vt1 - vt0;
		double3 vec2 = __GEIGEN__::__minus(vt2, vt0);
		double3 normal = __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(vec1, vec2));//vec1.cross(vec2).normalized();

		pos_normal_color[i * 9] = make_float3(vt0.x, vt0.y, vt0.z);
		pos_normal_color[i * 9 + 3] = make_float3(vt1.x, vt1.y, vt1.z);
		pos_normal_color[i * 9 + 6] = make_float3(vt2.x, vt2.y, vt2.z);

		pos_normal_color[i * 9 + 2] = make_float3(0.6875f, 0.51953f, 0.38671f);
		pos_normal_color[i * 9 + 5] = make_float3(0.6875f, 0.51953f, 0.38671f);
		pos_normal_color[i * 9 + 8] = make_float3(0.6875f, 0.51953f, 0.38671f);
		//}


		meshNormal[v0] = __GEIGEN__::__add(meshNormal[v0], normal);//normal;
		meshNormal[v1] = __GEIGEN__::__add(meshNormal[v1], normal);
		meshNormal[v2] = __GEIGEN__::__add(meshNormal[v2], normal);

	}
	for (int i = 0; i < number; i++)

	{
		int v0 = tetMesh.surface[i].x;
		int v1 = tetMesh.surface[i].y;
		int v2 = tetMesh.surface[i].z;
		//meshNormal[v0].normalize(); meshNormal[v1].normalize(); meshNormal[v2].normalize();
		pos_normal_color[i * 9 + 1] = make_float3(meshNormal[v0].x, meshNormal[v0].y, meshNormal[v0].z);
		pos_normal_color[i * 9 + 4] = make_float3(meshNormal[v1].x, meshNormal[v1].y, meshNormal[v1].z);
		pos_normal_color[i * 9 + 7] = make_float3(meshNormal[v2].x, meshNormal[v2].y, meshNormal[v2].z);
	}

	return pos_normal_color;
}


void draw_Scene3D() {
	//face.mesh3Ds[0] = mesh3d;
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glTranslatef(xTrans, yTrans, zTrans);
		glRotatef(xRot, 1.0f, 0.0f, 0.0f);
		glRotatef(yRot, 0.0f, 1.0f, 0.0f);

		draw_box3D(-1, -1, -1, 2, 2, 2, 1);
		if (drawSurface) {
			draw_mesh3D();
		}
		if (drawbvh) {
			draw_bvh();
		}

		glPopMatrix();


	glutSwapBuffers();
	//glFlush();
}
double mfsum = 0;
double total_time = 0;
int total_cg_iterations = 0;
int total_newton_iterations = 0;
int start = -1;

void saveScreenPic(const string& path) {
	std::stringstream ss;
	ss << path;
	ss.fill('0');
	ss.width(5);
	ss << step;
	std::string file_path = ss.str();

	SaveScreenShot(window_width, window_height, file_path);

}

void initFEM(tetrahedra_obj& mesh) {

	double massSum = 0;
	double volumeSum = 0;
	float angleX = FEM::PI / 4, angleY = -FEM::PI / 4, angleZ = FEM::PI / 2;
	__GEIGEN__::Matrix3x3d rotation, rotationZ, rotationY, rotationX, eigenTest;
	__GEIGEN__::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);
	__GEIGEN__::__set_Mat_val(rotationZ, cos(angleZ), -sin(angleZ), 0, sin(angleZ), cos(angleZ), 0, 0, 0, 1);
	__GEIGEN__::__set_Mat_val(rotationY, cos(angleY), 0, -sin(angleY), 0, 1, 0, sin(angleY), 0, cos(angleY));
	__GEIGEN__::__set_Mat_val(rotationX, 1, 0, 0, 0, cos(angleX), -sin(angleX), 0, sin(angleX), cos(angleX));

	//for (int j = 0; j < mesh.vertexNum; j++) {
	//    mesh.vertexes[j] = __GEIGEN__::__add(mesh.vertexes[j], make_double3(0.6, 0.6, 0.6));
	//}

	//for (int j = 0; j < mesh.vertexNum / 2; j++) {
	//    __GEIGEN__::Matrix3x3d rotate = __GEIGEN__::__M_Mat_multiply(rotationX, __GEIGEN__::__M_Mat_multiply(rotationY, rotationZ));
	//    mesh.vertexes[j] = __GEIGEN__::__add(__GEIGEN__::__M_v_multiply(rotate, __GEIGEN__::__add(mesh.vertexes[j], make_double3(0.0, -0.3, 0))), make_double3(0.0, 0.4, 0));
	//    mesh.vertexes[j] = __GEIGEN__::__add(mesh.vertexes[j], make_double3(0.0, 0.3, 0));
	//}
	//for (int j = mesh.vertexNum / 2; j < mesh.vertexNum; j++) {
	//    angleX = PI / 2, angleY = PI / 4, angleZ = PI / 2;
	//    __GEIGEN__::__set_Mat_val(rotationY, cos(angleY), 0, -sin(angleY), 0, 1, 0, sin(angleY), 0, cos(angleY));
	//    __GEIGEN__::__set_Mat_val(rotationX, 1, 0, 0, 0, cos(angleX), -sin(angleX), 0, sin(angleX), cos(angleX));
	//    __GEIGEN__::__set_Mat_val(rotationZ, cos(angleZ), -sin(angleZ), 0, sin(angleZ), cos(angleZ), 0, 0, 0, 1);

	//    __GEIGEN__::Matrix3x3d rotate = __GEIGEN__::__M_Mat_multiply(rotationX, __GEIGEN__::__M_Mat_multiply(rotationY, rotationZ));
	//    //mesh.vertexes[j] = __GEIGEN__::__add(__GEIGEN__::__M_v_multiply(rotate, __GEIGEN__::__add(mesh.vertexes[j], make_double3(0.0, 0.3, 0))), make_double3(0.0, -0.4, 0));
	//    __GEIGEN__::__init_Mat3x3(mesh.constraints[j], 0);
	//}

	//for (int j = 0; j < mesh.vertexNum; j++) {
	//    if ((mesh.vertexes[j].x) > 0.5-1e-4) {
	//        mesh.boundaryTypies[j] = 1;
	//        __GEIGEN__::__init_Mat3x3(mesh.constraints[j], 0);
	//    }
	//    if (((mesh.vertexes[j].x) < -0.5 + 1e-4)) {
	//        mesh.boundaryTypies[j] = -1;
	//        __GEIGEN__::__init_Mat3x3(mesh.constraints[j], 0);
	//    }
	//}

//   for (int j = 0; j < mesh.vertexNum; j++) {
//       double dis = pow(mesh.vertexes[j].y,2)+pow(mesh.vertexes[j].z,2);
//       if (dis<0.012*0.012&&mesh.vertexes[j].x>-0.25&&mesh.vertexes[j].x<-0.07) {
//           mesh.boundaryTypies[j] = 1;
//           __GEIGEN__::__init_Mat3x3(mesh.constraints[j], 0);
//       }
//   }


	for (int i = 0; i < mesh.tetrahedraNum; i++) {
		__GEIGEN__::Matrix3x3d DM;
		__calculateDms3D_double(mesh.vertexes.data(), mesh.tetrahedras[i], DM);//calculateDms3D_double(mesh.vertexes, mesh.tetrahedras[i], 0);

		__GEIGEN__::Matrix3x3d DM_inverse;
		__GEIGEN__::__Inverse(DM, DM_inverse);

		double vlm = calculateVolum(mesh.vertexes.data(), mesh.tetrahedras[i]);

		mesh.masses[mesh.tetrahedras[i].x] += vlm * ipc.density / 4;
		mesh.masses[mesh.tetrahedras[i].y] += vlm * ipc.density / 4;
		mesh.masses[mesh.tetrahedras[i].z] += vlm * ipc.density / 4;
		mesh.masses[mesh.tetrahedras[i].w] += vlm * ipc.density / 4;

		massSum += vlm * ipc.density;
		volumeSum += vlm;
		mesh.DM_inverse.push_back(DM_inverse);
		mesh.volum.push_back(vlm);

	}

	for (int i = 0; i < mesh.triangles.size(); i++) {
		__GEIGEN__::Matrix2x2d DM;
		__calculateDm2D_double(mesh.vertexes.data(), mesh.triangles[i], DM);

		__GEIGEN__::Matrix2x2d DM_inverse;
		__GEIGEN__::__Inverse2x2(DM, DM_inverse);

		double area = calculateArea(mesh.vertexes.data(), mesh.triangles[i]);
		area *= ipc.clothThickness;
		mesh.area.push_back(area);


		mesh.masses[mesh.triangles[i].x] += ipc.clothDensity * area / 3;
		mesh.masses[mesh.triangles[i].y] += ipc.clothDensity * area / 3;
		mesh.masses[mesh.triangles[i].z] += ipc.clothDensity * area / 3;

		massSum += area * ipc.clothDensity;
		volumeSum += area;
		mesh.tri_DM_inverse.push_back(DM_inverse);
	}

	mesh.meanMass = massSum / mesh.vertexNum;
	printf("meanMass: %f\n", mesh.meanMass);
	mesh.meanVolum = volumeSum / mesh.vertexNum;
}

void DefaultSettings()
{
	// global settings
	ipc.density = 1e3;
	ipc.YoungModulus = 1e5;
	ipc.PoissonRate = 0.49;
	ipc.lengthRateLame = ipc.YoungModulus / (2 * (1 + ipc.PoissonRate));
	ipc.volumeRateLame = ipc.YoungModulus * ipc.PoissonRate / ((1 + ipc.PoissonRate) * (1 - 2 * ipc.PoissonRate));
	ipc.lengthRate = 4 * ipc.lengthRateLame / 3;
	ipc.volumeRate = ipc.volumeRateLame + 5 * ipc.lengthRateLame / 6;
	ipc.frictionRate = 0.4;
	ipc.clothThickness = 1e-3;
	ipc.clothYoungModulus = 1e6;
	ipc.stretchStiff = ipc.clothYoungModulus / (2 * (1 + ipc.PoissonRate));
	ipc.shearStiff = ipc.stretchStiff * 0.05;
	ipc.clothDensity = 2e2;
	ipc.softMotionRate = 1e0;
	ipc.bendStiff = 3e-4;
	ipc.Newton_solver_threshold = 1e-1;
	ipc.pcg_threshold = 1e-3;
	ipc.IPC_dt = 1e-2;
	ipc.relative_dhat = 1e-3;
	ipc.bendStiff = ipc.clothYoungModulus * pow(ipc.clothThickness, 3) / (24 * (1 - ipc.PoissonRate * ipc.PoissonRate));
	ipc.shearStiff = 0.03 * ipc.stretchStiff;
}

void LoadSettings() {
	bool successfulRead = false;

	//read file
	std::ifstream infile;

#ifdef _WIN32
	string DEFAULT_CONFIG_FILE = std::string{ gipc::assets_dir() } + "scene/windows/parameterSetting.txt";
#else
	string DEFAULT_CONFIG_FILE = "../GPU_IPC/mLBVH/scene/parameterSetting.txt";
#endif


	infile.open(DEFAULT_CONFIG_FILE, std::ifstream::in);
	if (successfulRead = infile.is_open())
	{
		int tempEnum;
		char ignoreToken[256];

		// global settings:
		infile >> ignoreToken >> ipc.density;
		infile >> ignoreToken >> ipc.YoungModulus;
		infile >> ignoreToken >> ipc.PoissonRate;
		infile >> ignoreToken >> ipc.frictionRate;
		infile >> ignoreToken >> ipc.clothThickness;
		infile >> ignoreToken >> ipc.clothYoungModulus;
		//infile >> ignoreToken >> ipc.shearStiff;
		infile >> ignoreToken >> ipc.clothDensity;
		infile >> ignoreToken >> ipc.softMotionRate;
		//infile >> ignoreToken >> ipc.bendStiff;
		infile >> ignoreToken >> collision_detection_buff_scale;
		infile >> ignoreToken >> motion_rate;
		infile >> ignoreToken >> ipc.IPC_dt;
		infile >> ignoreToken >> ipc.pcg_threshold;
		infile >> ignoreToken >> ipc.Newton_solver_threshold;
		infile >> ignoreToken >> ipc.relative_dhat;


		ipc.lengthRateLame = ipc.YoungModulus / (2 * (1 + ipc.PoissonRate));
		ipc.volumeRateLame = ipc.YoungModulus * ipc.PoissonRate / ((1 + ipc.PoissonRate) * (1 - 2 * ipc.PoissonRate));
		ipc.lengthRate = 4 * ipc.lengthRateLame / 3;
		ipc.volumeRate = ipc.volumeRateLame + 5 * ipc.lengthRateLame / 6;
		ipc.stretchStiff = ipc.clothYoungModulus / (2 * (1 + ipc.PoissonRate));

		ipc.bendStiff = ipc.clothYoungModulus * pow(ipc.clothThickness, 3) / (24 * (1 - ipc.PoissonRate * ipc.PoissonRate));
		ipc.shearStiff = 0.03 * ipc.stretchStiff;
		printf("ipc.bendStiff: %f\n", ipc.bendStiff);
		//ipc.shearStiff = 
		infile.close();
	}

	if (!successfulRead)
	{
		std::cerr << "Waning: failed loading settings, set to defaults." << std::endl;
		DefaultSettings();
	}
}



void initScene1() {

	auto assets_dir = std::string{ gipc::assets_dir() };

    tetMesh.load_tetrahedraMesh(
        assets_dir + "tetMesh/bunny.msh", 0.5, make_double3(0, 0.4, 0));
    tetMesh.load_tetrahedraMesh(
        assets_dir + "tetMesh/bunny.msh", 0.5, make_double3(0, -0.4, 0));


	tetMesh.getSurface();

	initFEM(tetMesh);
	//device_TetraData d_tetMesh;
	d_tetMesh.Malloc_DEVICE_MEM(tetMesh.vertexNum, tetMesh.tetrahedraNum, tetMesh.triangleNum, tetMesh.softNum, tetMesh.tri_edges.size());

	CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.masses, tetMesh.masses.data(), tetMesh.vertexNum * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.volum, tetMesh.volum.data(), tetMesh.tetrahedraNum * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.vertexes, tetMesh.vertexes.data(), tetMesh.vertexNum * sizeof(double3), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.o_vertexes, tetMesh.vertexes.data(), tetMesh.vertexNum * sizeof(double3), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.tetrahedras, tetMesh.tetrahedras.data(), tetMesh.tetrahedraNum * sizeof(uint4), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.DmInverses, tetMesh.DM_inverse.data(), tetMesh.tetrahedraNum * sizeof(__GEIGEN__::Matrix3x3d), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.Constraints, tetMesh.constraints.data(), tetMesh.vertexNum * sizeof(__GEIGEN__::Matrix3x3d), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.BoundaryType, tetMesh.boundaryTypies.data(), tetMesh.vertexNum * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.velocities, tetMesh.velocities.data(), tetMesh.vertexNum * sizeof(double3), cudaMemcpyHostToDevice));


	CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.targetIndex, tetMesh.targetIndex.data(), tetMesh.softNum * sizeof(uint32_t), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.targetVert, tetMesh.targetPos.data(), tetMesh.softNum * sizeof(double3), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.triDmInverses, tetMesh.tri_DM_inverse.data(), tetMesh.triangleNum * sizeof(__GEIGEN__::Matrix2x2d), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.area, tetMesh.area.data(), tetMesh.triangleNum * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.triangles, tetMesh.triangles.data(), tetMesh.triangleNum * sizeof(uint3), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.tri_edges, tetMesh.tri_edges.data(), tetMesh.tri_edges.size() * sizeof(uint2), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.tri_edge_adj_vertex, tetMesh.tri_edges_adj_points.data(), tetMesh.tri_edges.size() * sizeof(uint2), cudaMemcpyHostToDevice));




	printf("stretchStiff:  %f,  shearStiff:   %f\n", ipc.stretchStiff, ipc.shearStiff);

	ipc.vertexNum = tetMesh.vertexNum;
	ipc.tetrahedraNum = tetMesh.tetrahedraNum;
	ipc._vertexes = d_tetMesh.vertexes;
	ipc._rest_vertexes = d_tetMesh.rest_vertexes;
	ipc.surf_vertexNum = tetMesh.surfVerts.size();
	ipc.surface_Num = tetMesh.surface.size();
	ipc.edge_Num = tetMesh.surfEdges.size();
	ipc.tri_edge_num = tetMesh.tri_edges.size();

	//ipc.IPC_dt = 0.01 / 1.0;//1.0 / 30;//1.0 / 100;
	ipc.MAX_CCD_COLLITION_PAIRS_NUM = 1 * collision_detection_buff_scale * (((double)(ipc.surface_Num * 15 + ipc.edge_Num * 10)) * std::max((ipc.IPC_dt / 0.01), 2.0));
	ipc.MAX_COLLITION_PAIRS_NUM = (ipc.surf_vertexNum * 3 + ipc.edge_Num * 2) * 3 * collision_detection_buff_scale;

	ipc.triangleNum = tetMesh.triangleNum;
	ipc.targetVert = d_tetMesh.targetVert;
	ipc.targetInd = d_tetMesh.targetIndex;
	ipc.softNum = tetMesh.softNum;


	printf("vertNum: %d      tetraNum: %d      faceNum: %d\n", ipc.vertexNum, ipc.tetrahedraNum, ipc.surface_Num);
	printf("surfVertNum: %d      surfEdgesNum: %d\n", ipc.surf_vertexNum, ipc.edge_Num);
	printf("maxCollisionPairsNum_CCD: %d      maxCollisionPairsNum: %d\n", ipc.MAX_CCD_COLLITION_PAIRS_NUM, ipc.MAX_COLLITION_PAIRS_NUM);

	//ipc.USE_MAS = false;
	ipc.pcg_data.P_type = 0;
	ipc.MALLOC_DEVICE_MEM();

	CUDA_SAFE_CALL(cudaMemcpy(ipc._faces, tetMesh.surface.data(), ipc.surface_Num * sizeof(uint3), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(ipc._edges, tetMesh.surfEdges.data(), ipc.edge_Num * sizeof(uint2), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(ipc._surfVerts, tetMesh.surfVerts.data(), ipc.surf_vertexNum * sizeof(uint32_t), cudaMemcpyHostToDevice));
	ipc.initBVH(d_tetMesh.BoundaryType);

	if (ipc.pcg_data.P_type) {
		int neighborListSize = tetMesh.getVertNeighbors();
		ipc.pcg_data.MP.initPreconditioner(ipc.vertexNum, neighborListSize, ipc._collisonPairs);

		ipc.pcg_data.MP.neighborListSize = neighborListSize;
		CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborListInit, tetMesh.neighborList.data(), neighborListSize * sizeof(unsigned int), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborStart, tetMesh.neighborStart.data(), ipc.vertexNum * sizeof(unsigned int), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborNumInit, tetMesh.neighborNum.data(), ipc.vertexNum * sizeof(unsigned int), cudaMemcpyHostToDevice));

	}


	if (true) {
		if (ipc.animation) {
			ipc.sortMesh(d_tetMesh, bodyVertOffset);
		}
		else {
			ipc.sortMesh(d_tetMesh, ipc.vertexNum);
		}
		CUDA_SAFE_CALL(cudaMemcpy(tetMesh.vertexes.data(), d_tetMesh.vertexes, tetMesh.vertexNum * sizeof(double3), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(tetMesh.surface.data(), ipc._faces, ipc.surface_Num * sizeof(uint3), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(tetMesh.surfEdges.data(), ipc._edges, ipc.edge_Num * sizeof(uint2), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(tetMesh.surfVerts.data(), ipc._surfVerts, ipc.surf_vertexNum * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	}
	CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.rest_vertexes, d_tetMesh.o_vertexes, ipc.vertexNum * sizeof(double3), cudaMemcpyDeviceToDevice));

	ipc.buildBVH();
	ipc.init(tetMesh.meanMass, tetMesh.meanVolum, tetMesh.minConer, tetMesh.maxConer);
	printf("bboxDiagSize2: %f\n", ipc.bboxDiagSize2);
	printf("maxConer: %f  %f   %f           minCorner: %f  %f   %f\n", tetMesh.maxConer.x, tetMesh.maxConer.y, tetMesh.maxConer.z, tetMesh.minConer.x, tetMesh.minConer.y, tetMesh.minConer.z);

	printf("restSNKE: %f\n", ipc.RestNHEnergy);
	ipc.buildCP();
	ipc.pcg_data.b = d_tetMesh.fb;
	ipc._moveDir = ipc.pcg_data.dx;
	ipc.animation_subRate = 1.0 / motion_rate;
	//ipc.animation_fullRate = ipc.animation_subRate;
	ipc.computeXTilta(d_tetMesh, 1);
	///////////////////////////////////////////////////////////////////////////////////

	//if (ipc.isIntersected(d_tetMesh)) {
	//    printf("init intersection\n");
	//}

	bvs.resize(2 * ipc.edge_Num - 1);
	nodes.resize(2 * ipc.edge_Num - 1);
	//CUDA_SAFE_CALL(cudaDeviceSynchronize());
	CUDA_SAFE_CALL(cudaMemcpy(&bvs[0], ipc.bvh_e._bvs, (2 * ipc.edge_Num - 1) * sizeof(AABB), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(&nodes[0], ipc.bvh_e._nodes, (2 * ipc.edge_Num - 1) * sizeof(Node), cudaMemcpyDeviceToHost));
}


void display(void)
{
	draw_Scene3D();

	//    if (saveSurface) {
	//        saveSurfaceMesh("saveSurface/surf_");
	//        saveSurface = !saveSurface;
	//    }

	if (stop) return;

	ipc.IPC_Solver(d_tetMesh);

	//if(step == 20) {
	//    std::string filename = "triMesh/body/body_obj_1.obj";
	//    frameId++;
	//    tetMesh.load_animation(filename, 1, make_double3(0, 0, -0.0));
	//    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.targetVert, tetMesh.targetPos.data(), tetMesh.softNum * sizeof(double3), cudaMemcpyHostToDevice));
	//}

	CUDA_SAFE_CALL(cudaMemcpy(&bvs[0], ipc.bvh_e._bvs, (2 * ipc.edge_Num - 1) * sizeof(AABB), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(&nodes[0], ipc.bvh_e._nodes, (2 * ipc.edge_Num - 1) * sizeof(Node), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(tetMesh.vertexes.data(), ipc._vertexes, ipc.vertexNum * sizeof(double3), cudaMemcpyDeviceToHost));



	if (screenshot)
	{
		std::stringstream ss;
		ss << "saveScreen/step_";
		ss.fill('0');
		ss.width(5);
		ss << step / 1;
		std::string file_path = ss.str();
		SaveScreenShot(window_width, window_height, file_path);

	}
	step++;
	if (saveSurface) {
		saveSurfaceMesh("saveSurface/surf_");
		//saveSurface = !saveSurface;
	}

//    if(step>=60){
//        exit(0);
//    }
}

void init(void)
{
	Init_CUDA();

	//main2();

	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		/* Problem: glewInit failed, something is seriously wrong. */
		std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
	}
	std::cerr << "Status: Using GLEW " << glewGetString(GLEW_VERSION) << std::endl;
	glClearColor(0.0, 0.0, 0.0, 1.0);


	LoadSettings();
	initScene1();

		glViewport(0, 0, window_width, window_height);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(45.0, (float)window_width / window_height, 10.1f, 500.0);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glTranslatef(0.0f, 0.0f, -3.0f);

	//glEnable(GL_DEPTH_TEST);



}



void idle_func()
{
	glutPostRedisplay();
}

void reshape_func(GLint width, GLint height)
{
	//window_width = width;
	//window_height = height;

	glViewport(0, 0, width, height);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();

		gluPerspective(45.0, (float)width / height, 0.1, 500.0);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glTranslatef(0.0f, 0.0f, -3.0f);
	//glTranslatef(0.5f, 0.5f, -4.0f);
}

void keyboard_func(unsigned char key, int x, int y)
{
	if (key == 'w')
	{
		zTrans += .01f;
	}

	if (key == 's')
	{
		zTrans -= .01f;
	}

	if (key == 'a')
	{
		xTrans += .01f;
	}

	if (key == 'd')
	{
		xTrans -= .01f;
	}

	if (key == 'q')
	{
		yTrans -= .01f;
	}

	if (key == 'e')
	{
		yTrans += .01f;
	}

	if (key == '/')
	{
		screenshot = !screenshot;
	}

	if (key == '9')
	{
		saveSurface = !saveSurface;
	}

	if (key == 'k')
	{
		drawSurface = !drawSurface;
	}

	if (key == 'f')
	{
		drawbvh = !drawbvh;
	}

	if (key == ' ')
	{
		stop = !stop;
	}
	glutPostRedisplay();
}

void special_keyboard_func(int key, int x, int y)
{
	glutPostRedisplay();
}

void mouse_func(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		buttonState = 1;
	}
	else if (state == GLUT_UP)
	{
		buttonState = 0;
	}

	ox = x; oy = y;

	glutPostRedisplay();
}

void motion_func(int x, int y)
{
	float dx, dy;
	dx = (float)(x - ox);
	dy = (float)(y - oy);

	if (buttonState == 1)
	{
		xRot += dy / 5.0f;
		yRot += dx / 5.0f;
	}

	ox = x; oy = y;

	glutPostRedisplay();
}


void SpecialKey(GLint key, GLint x, GLint y)
{
	if (key == GLUT_KEY_DOWN)
	{
		change = true;
		initPath -= 1;
		if (initPath < 0) {
			initPath = obj_pathes.size() - 1;
		}
	}

	if (key == GLUT_KEY_UP)
	{
		change = true;
		initPath += 1;
		if (initPath == obj_pathes.size()) {
			initPath = 0;
		}
	}
	glutPostRedisplay();
}


int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	//glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);

	glutSetOption(GLUT_MULTISAMPLE, 16);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_MULTISAMPLE);

	glutInitWindowSize(window_width, window_height);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("FEM");

	init();

	//glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
	//glEnable(GL_POINT_SPRITE_ARB);
	//glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);


	glEnable(GL_MULTISAMPLE);
	glHint(GL_MULTISAMPLE_FILTER_HINT_NV, GL_NICEST);


	glutDisplayFunc(display);


	//glutDisplayFunc(display_func);
	glutReshapeFunc(reshape_func);
	glutKeyboardFunc(keyboard_func);
	glutSpecialFunc(&SpecialKey);
	glutMouseFunc(mouse_func);
	glutMotionFunc(motion_func);
	glutIdleFunc(idle_func);


	glutMainLoop();
	//return 0;
}

