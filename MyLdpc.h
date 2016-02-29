/*
 * MyLdpc.h
 *
 *  Created on: Feb 3, 2016
 *      Author: wing
 */

#ifndef MYLDPC_H_
#define MYLDPC_H_

#include <vector>
#include<iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include"cl.hpp"
//using namespace Eigen;

#define LDPC_SUCCESS	0
#define LDPC_FAIL		0
#define MAX				200000
#define DataType		int
#define LOG			std::cout<<__FILE__<<" "<<__LINE__<<std::endl
#define LOGA(a)			std::cout<<__FILE__<<" "<<__LINE__<<" "<<a<<std::endl
#define ASSERT(a)		if(!(a)) {LOG;std::cout<<"Assert! error";exit(0);}
#define GETTIME			LOGA((double)clock()/CLOCKS_PER_SEC)

enum rate_type {
	rate_1_2, rate_2_3_a, rate_2_3_b, rate_3_4_a, rate_3_4_b, rate_5_6
};

enum decodeType {
	DecodeCPU, DecodeMS, DecodeSP
};
const char h_seed_1_2[] = { -1, 94, 73, -1, -1, -1, -1, -1, 55, 83, -1, -1, 7,
		0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 27, -1, -1, -1, 22, 79,
		9, -1, -1, -1, 12, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		-1, 24, 22, 81, -1, 33, -1, -1, -1, 0, -1, -1, 0, 0, -1, -1, -1, -1, -1,
		-1, -1, -1, 61, -1, 47, -1, -1, -1, -1, -1, 65, 25, -1, -1, -1, -1, -1,
		0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 39, -1, -1, -1, 84, -1, -1,
		41, 72, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		-1, 46, 40, -1, 82, -1, -1, -1, 79, 0, -1, -1, -1, -1, 0, 0, -1, -1, -1,
		-1, -1, -1, -1, 95, 53, -1, -1, -1, -1, -1, 14, 18, -1, -1, -1, -1, -1,
		-1, -1, 0, 0, -1, -1, -1, -1, -1, 11, 73, -1, -1, -1, 2, -1, -1, 47, -1,
		-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, 12, -1, -1, -1, 83,
		24, -1, 43, -1, -1, -1, 51, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1,
		-1, -1, -1, -1, -1, -1, 94, -1, 59, -1, -1, 70, 72, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, 0, 0, -1, -1, -1, 7, 65, -1, -1, -1, -1, 39, 49, -1, -1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 43, -1, -1, -1, -1, 66,
		-1, 41, -1, -1, -1, 26, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0 };
const char h_seed_2_3_a[] = { 3, 0, -1, -1, 2, 0, -1, 3, 7, -1, 1, 1, -1, -1,
		-1, -1, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 36, -1, -1, 34, 10,
		-1, -1, 18, 2, -1, 3, 0, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, 12, 2,
		-1, 15, -1, 40, -1, 3, -1, 15, -1, 2, 13, -1, -1, -1, 0, 0, -1, -1, -1,
		-1, -1, -1, 19, 24, -1, 3, 0, -1, 6, -1, 17, -1, -1, -1, 8, 39, -1, -1,
		-1, 0, 0, -1, -1, -1, 20, -1, 6, -1, -1, 10, 29, -1, -1, 28, -1, 14, -1,
		38, -1, -1, 0, -1, -1, -1, 0, 0, -1, -1, -1, -1, 10, -1, 28, 20, -1, -1,
		8, -1, 36, -1, 9, -1, 21, 45, -1, -1, -1, -1, -1, 0, 0, -1, 35, 25, -1,
		37, -1, 21, -1, -1, 5, -1, -1, 0, -1, 4, 20, -1, -1, -1, -1, -1, -1, -1,
		0, 0, -1, 6, 6, -1, -1, -1, 4, -1, 14, 30, -1, 3, 36, -1, 14, -1, 1, -1,
		-1, -1, -1, -1, -1, 0 };
const char h_seed_2_3_b[] = { 2, -1, 19, -1, 47, -1, 48, -1, 36, -1, 82, -1, 47,
		-1, 15, -1, 95, 0, -1, -1, -1, -1, -1, -1, -1, 69, -1, 88, -1, 33, -1,
		3, -1, 16, -1, 37, -1, 40, -1, 48, -1, 0, 0, -1, -1, -1, -1, -1, 10, -1,
		86, -1, 62, -1, 28, -1, 85, -1, 16, -1, 34, -1, 73, -1, -1, -1, 0, 0,
		-1, -1, -1, -1, -1, 28, -1, 32, -1, 81, -1, 27, -1, 88, -1, 5, -1, 56,
		-1, 37, -1, -1, -1, 0, 0, -1, -1, -1, 23, -1, 29, -1, 15, -1, 30, -1,
		66, -1, 24, -1, 50, -1, 62, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, 30,
		-1, 65, -1, 54, -1, 14, -1, 0, -1, 30, -1, 74, -1, 0, -1, -1, -1, -1,
		-1, 0, 0, -1, 32, -1, 0, -1, 15, -1, 56, -1, 85, -1, 5, -1, 6, -1, 52,
		-1, 0, -1, -1, -1, -1, -1, 0, 0, -1, 0, -1, 47, -1, 13, -1, 61, -1, 84,
		-1, 55, -1, 78, -1, 41, 95, -1, -1, -1, -1, -1, -1, 0 };
const char h_seed_3_4_a[] = { 6, 38, 3, 93, -1, -1, -1, 30, 70, -1, 86, -1, 37,
		38, 4, 11, -1, 46, 48, 0, -1, -1, -1, -1, 62, 94, 19, 84, -1, 92, 78,
		-1, 15, -1, -1, 92, -1, 45, 24, 32, 30, -1, -1, 0, 0, -1, -1, -1, 71,
		-1, 55, -1, 12, 66, 45, 79, -1, 78, -1, -1, 10, -1, 22, 55, 70, 82, -1,
		-1, 0, 0, -1, -1, 38, 61, -1, 66, 9, 73, 47, 64, -1, 39, 61, 43, -1, -1,
		-1, -1, 95, 32, 0, -1, -1, 0, 0, -1, -1, -1, -1, -1, 32, 52, 55, 80, 95,
		22, 6, 51, 24, 90, 44, 20, -1, -1, -1, -1, -1, -1, 0, 0, -1, 63, 31, 88,
		20, -1, -1, -1, 6, 40, 56, 16, 71, 53, -1, -1, 27, 26, 48, -1, -1, -1,
		-1, 0 };
const char h_seed_3_4_b[] = { -1, 81, -1, 28, -1, -1, 14, 25, 17, -1, -1, 85,
		29, 52, 78, 95, 22, 92, 0, 0, -1, -1, -1, -1, 42, -1, 14, 68, 32, -1,
		-1, -1, -1, 70, 43, 11, 36, 40, 33, 57, 38, 24, -1, 0, 0, -1, -1, -1,
		-1, -1, 20, -1, -1, 63, 39, -1, 70, 67, -1, 38, 4, 72, 47, 29, 60, 5,
		80, -1, 0, 0, -1, -1, 64, 2, -1, -1, 63, -1, -1, 3, 51, -1, 81, 15, 94,
		9, 85, 36, 14, 19, -1, -1, -1, 0, 0, -1, -1, 53, 60, 80, -1, 26, 75, -1,
		-1, -1, -1, 86, 77, 1, 3, 72, 60, 25, -1, -1, -1, -1, 0, 0, 77, -1, -1,
		-1, 15, 28, -1, 35, -1, 72, 30, 68, 85, 84, 26, 64, 11, 89, 0, -1, -1,
		-1, -1, 0 };
const char h_seed_5_6[] = { 1, 25, 55, -1, 47, 4, -1, 91, 84, 8, 86, 52, 82, 33,
		5, 0, 36, 20, 4, 77, 80, 0, -1, -1, -1, 6, -1, 36, 40, 47, 12, 79, 47,
		-1, 41, 21, 12, 71, 14, 72, 0, 44, 49, 0, 0, 0, 0, -1, 51, 81, 83, 4,
		67, -1, 21, -1, 31, 24, 91, 61, 81, 9, 86, 78, 60, 88, 67, 15, -1, -1,
		0, 0, 50, -1, 50, 15, -1, 36, 13, 10, 11, 20, 53, 90, 29, 92, 57, 30,
		84, 92, 11, 66, 80, -1, -1, 0 };
const int n_b = 24;

class Coder {
public:

	Coder(int ldpcK, int ldpcN, enum rate_type rate);
	~Coder();
	int forEncoder();
	int forDecoder(int batchSize);
	int addDecodeType(enum decodeType deType);
	int forTest();

	int encode(char * srcCode, char * priorCode, int srcLength);
	//code length in decode is 8 times of code length in encode;
	int decode(float * postCode, char * srcCode, int srcLength,
			enum decodeType deType);

	int test(char* priorCode, float * postCode, int priorCodeLength,
			float rate);
	//int testMS(char* priorCode, float * postCode, int priorCodeLength,
	//float rate);

	int getPriorCodeLength(int srcLength);
	int getPostCodeLength(int srcLength);
	int getCodeSize(int srcLength);

	Eigen::SparseMatrix<DataType> checkMatrix;
private:
	double stepTime[10];
	int initCheckMatrix();
	int encodeOnce(char * src, char * code, int srcLength);
	int decodeOnceSP(float * postCode, char * src, int postCodeLength,
			int srcLength);
	int decodeOnceMS(float * postCode, char * src, int postCodeLength,
			int srcLength);
	int decodeNoCL(float * postCode, char * srcCode, int srcLength);
	int decodeCPU(float * postCode, char * srcCode, int srcLength);
	int srcLength;
	int codeLength;
	int ldpcK;
	int ldpcN;
	int ldpcM;
	int z;
	int nonZeros;
	int batchSize;

	bool isEncoder, isDecoder,isDecodeMS,isDecodeSP;

	bool* flagsBool;bool* srcBool;bool* isDones;

	//int *outputSrcInt;

	int * hColFirstPtr;
	int * hColNextPtr;
	int * hRowFirstPtr;
	int * hRowNextPtr;
	int * hCols;
	int * hRows;
	enum rate_type rate;

	Eigen::SparseMatrix<DataType> smInvT;
	Eigen::SparseMatrix<DataType> smA;
	Eigen::SparseMatrix<DataType> smB;
	Eigen::SparseMatrix<DataType> smTmp;
	Eigen::SparseMatrix<DataType> smK;
	Eigen::SparseMatrix<DataType> smP1;
	Eigen::SparseMatrix<DataType> smP2;

	cl_int errNum;
	cl::Context context;
	cl::CommandQueue queue;
	cl::Program program;
	cl::Buffer memHRow;
	cl::Buffer memHCol;

	cl::Buffer memR0;
	cl::Buffer memR1;
	cl::Buffer memQ0;
	cl::Buffer memQ1;

	cl::Buffer memPriorP0;
	cl::Buffer memPriorP1;

	cl::Buffer memHRowFirstPtr;
	cl::Buffer memHRowNextPtr;
	cl::Buffer memHColFirstPtr;
	cl::Buffer memHColNextPtr;

	cl::Buffer memCodes;

	cl::Buffer memLPostP;
	cl::Buffer memLR;
	cl::Buffer memLQA;
	cl::Buffer memLQB;

	cl::Buffer memSrcBool;
	cl::Buffer memFlagsBool;
	cl::Buffer memSrcCode;
	cl::Buffer memIsDones;

	//create kernel
	cl::Kernel kerDecodeInit;
	cl::Kernel kerRefreshR;
	cl::Kernel kerRefreshQ;
	cl::Kernel kerHardDecision;
	cl::Kernel kerCheckResult;

	cl::Kernel kerDecodeInitMS;
	cl::Kernel kerRefreshRMS;
	cl::Kernel kerRefreshQMS;
	cl::Kernel kerRefreshPostPMS;
	cl::Kernel kerCheckResultMS;
	cl::Kernel kerToChar;

};

namespace Eigen {
template<class Dtype>
Matrix<Dtype, Dynamic, Dynamic> inverse(Matrix<Dtype, Dynamic, Dynamic> matSrc);
template<class Dtype>
void binary(Matrix<Dtype, Dynamic, Dynamic> & mat);
template<class Dtype>
void binarySM(SparseMatrix<Dtype> & sm);
template<class Dtype>
SparseMatrix<Dtype> dense2Sparse(const Matrix<Dtype, Dynamic, Dynamic> &dm);

template<class Dtype>
Matrix<Dtype, Dynamic, Dynamic> inverse(
		Matrix<Dtype, Dynamic, Dynamic> matSrc) {
	int size = matSrc.rows();
	if (matSrc.rows() != matSrc.cols()) {
		printf("Error,row != col, no inverse");
		exit(0);
	}
	Matrix<Dtype, Dynamic, Dynamic> mat;  //=new Matrix<int, Dynamic,Dynamic>();
	Matrix<Dtype, Dynamic, Dynamic> inv;  //=new Matrix<int, Dynamic,Dynamic>();
	mat = matSrc;
	binary(mat);
	inv.resize(size, size);
	for (int r = 0; r < size; ++r) {
		for (int c = 0; c < size; ++c) {
			if (r == c)
				inv(r, c) = 1;
			else
				inv(r, c) = 0;
		}
	}
	for (int c = 0; c < size; ++c) {
		for (int r = 0; r < size; ++r) {
			if (mat(r, c) != 0) {
				Dtype tmp = mat(r, c);
				mat.col(c) += (1 / tmp - 1) * mat.col(c);
				inv.col(c) += (1 / tmp - 1) * inv.col(c);
				for (int c2 = c + 1; c2 < size; ++c2) {
					mat.col(c2) -= mat(r, c2) * mat.col(c);
					inv.col(c2) -= mat(r, c2) * inv.col(c);
				}
				break;
			}
		}
	}
	for (int r = 0; r < size; r++) {
		for (int r2 = r + 1; r2 < size; r2++) {
			Dtype tmp = mat(r2, r);
			mat.row(r2) -= tmp * mat.row(r);
			inv.row(r2) -= tmp * inv.row(r);
		}
	}
	binary(inv);
	return inv;
}

template<class Dtype>
void binary(Matrix<Dtype, Dynamic, Dynamic> & mat) {
	for (int r = 0; r < mat.rows(); ++r) {
		for (int c = 0; c < mat.cols(); ++c) {
			if (mat(r, c) < 0)
				mat(r, c) = (-mat(r, c)) % 2;
			else
				mat(r, c) = mat(r, c) % 2;
		}
	}
}

template<class Dtype>
void binarySM(SparseMatrix<Dtype> & sm) {
	for (int k = 0; k < sm.outerSize(); ++k)
		for (SparseMatrix<DataType>::InnerIterator it(sm, k); it; ++it) {
			if (it.value() < 0)
				it.valueRef() = -(it.value()) % 2;
			else
				it.valueRef() = it.value() % 2;
		}
}

template<class Dtype>
SparseMatrix<Dtype> dense2Sparse(const Matrix<Dtype, Dynamic, Dynamic> &dm) {
	typedef Triplet<Dtype> T;
	const int rows = dm.rows();
	const int cols = dm.cols();
	SparseMatrix<Dtype> sm(rows, cols);
	std::vector<T> tripletList;

	for (int row = 0; row < rows; ++row) {
		for (int col = 0; col < cols; ++col) {
			if (dm(row, col)) {
				tripletList.push_back(T(row, col, 1));
			}
		}
	}
	sm.setFromTriplets(tripletList.begin(), tripletList.end());
	return sm;
}
}
char * load_program_source(const char *filename);
float gaussian(float ave, float VAR);

const char* openclErr2Str(cl_int error);

#endif /* MYLDPC_H_ */
