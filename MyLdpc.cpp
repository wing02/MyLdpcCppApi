//============================================================================
// Name        : MyLdpc.cpp
// Author      : wing
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C, Ansi-style
//============================================================================

#include "MyLdpc.h"

#include<iostream>
#include<cmath>
#include<math.h>
#include<stdlib.h>
#include<vector>
#include<stdlib.h>
#include<sys/stat.h>
#include<time.h>

Coder::Coder(int ldpcK, int ldpcN, enum rate_type rate) :
		ldpcK(ldpcK), ldpcN(ldpcN), ldpcM(ldpcN - ldpcK), rate(rate), checkMatrix(
				ldpcN - ldpcK, ldpcN), isDecoder(false), isEncoder(false), isDecodeMS(
		false), isDecodeSP(false) {
	times = 40;
	initCheckMatrix();
	for (int i = 0; i < 10; ++i) {
		stepTime[i] = 0;
	}
}

Coder::~Coder() {
	using namespace std;
	if (isDecoder) {
		free(hRows);
		free(hCols);
		free(hRowFirstPtr);
		free(hRowNextPtr);
		free(hColFirstPtr);
		free(hColNextPtr);
		free(flagsBool);
		free(srcBool);
		free(hRowRange);
	}
	if (isDecodeMS) {

	}
	if (isDecodeSP) {

	}

}
int Coder::initCheckMatrix() {
	using namespace Eigen;
	//he expansion factor (z factor) is equal to n/24 for priorCode length n;
	z = ldpcN / n_b;
	int seedRowLength;
	const int seedColLength = n_b;
	switch (rate) {
	case rate_1_2:
		hSeed = h_seed_1_2;
		seedRowLength = 12;
		break;
	case rate_2_3_a:
		hSeed = h_seed_2_3_a;
		seedRowLength = 8;
		break;
	case rate_2_3_b:
		hSeed = h_seed_2_3_b;
		seedRowLength = 8;
		break;
	case rate_3_4_a:
		hSeed = h_seed_3_4_a;
		seedRowLength = 6;
		break;
	case rate_3_4_b:
		hSeed = h_seed_3_4_b;
		seedRowLength = 6;
		break;
	case rate_5_6:
		hSeed = h_seed_5_6;
		seedRowLength = 4;
		break;
	}
	int permut;
	typedef Triplet<DataType> T;
	std::vector<T> tripletList;
	for (int seedRow = 0; seedRow < seedRowLength; ++seedRow) {
		for (int seedCol = 0; seedCol < seedColLength; ++seedCol) {
			if ((permut = hSeed[seedRow * seedColLength + seedCol]) >= 0) {
				if (rate != rate_2_3_a) {
					permut = permut * z / 96;
				} else {
					permut = permut % z;
				}
				for (int permutRow = 0; permutRow < z; ++permutRow) {
					for (int permutCol = 0; permutCol < z; ++permutCol) {
						if ((z + permutCol - permutRow) % z == permut) {
							tripletList.push_back(
									T(seedRow * z + permutRow,
											seedCol * z + permutCol, 1));
						}
					}
				}

			}
		}
	}
	checkMatrix.setFromTriplets(tripletList.begin(), tripletList.end());
	nonZeros = checkMatrix.nonZeros();

	//get max col weight;
	maxColWt = 0;
	maxRowWt = 0;
	int tmp;
	for (int seedRow = 0; seedRow < seedRowLength; ++seedRow) {
		tmp = 0;
		for (int seedCol = 0; seedCol < seedColLength; ++seedCol) {
			if (hSeed[seedRow * seedColLength + seedCol] != -1)
				++tmp;
		}
		if (tmp > maxColWt)
			maxColWt = tmp;
	}
	for (int seedCol = 0; seedCol < seedColLength; ++seedCol) {
		tmp = 0;
		for (int seedRow = 0; seedRow < seedRowLength; ++seedRow) {
			if (hSeed[seedRow * seedColLength + seedCol] != -1)
				++tmp;
		}
		if (tmp > maxRowWt)
			maxRowWt = tmp;
	}

	return LDPC_SUCCESS;
}

int Coder::forEncoder() {
	isEncoder = true;
	using namespace Eigen;
	//g=z
	typedef Matrix<DataType, Dynamic, Dynamic> DenseMatrix;
	DenseMatrix A = checkMatrix.block(0, 0, ldpcM - z, ldpcK);
	DenseMatrix B = checkMatrix.block(0, ldpcK, ldpcM - z, z);
	DenseMatrix C = checkMatrix.block(ldpcM - z, 0, z, ldpcK);
	DenseMatrix D = checkMatrix.block(ldpcM - z, ldpcK, z, z);
	DenseMatrix T = checkMatrix.block(0, ldpcK + z, ldpcM - z, ldpcM - z);
	DenseMatrix E = checkMatrix.block(ldpcM - z, ldpcK + z, z, ldpcM - z);

	DenseMatrix invT = inverse(T);
	DenseMatrix EinvT = E * invT;
	DenseMatrix EinvTBaddD = -EinvT * B + D;
	//DenseMatrix tmp = (-EinvT * A + C);
	DenseMatrix tmp = inverse(EinvTBaddD) * (-EinvT * A + C);
	binary(tmp);

	smInvT = dense2Sparse(invT);
	smA = dense2Sparse(A);
	smB = dense2Sparse(B);
	smTmp = dense2Sparse(tmp);
	smK.resize(ldpcK, 1);
	smP1.resize(z, 1);
	smP2.resize(ldpcM - z, 1);

	return LDPC_SUCCESS;
}

int Coder::forDecoder(int batchSize) {
	isDecoder = true;
	using namespace Eigen;
	this->batchSize = batchSize;
	//create a  adjacency links for ColPtr;
	hColFirstPtr = (int *) malloc(ldpcN * sizeof(int));
	hColNextPtr = (int *) malloc(nonZeros * sizeof(int));
	memset(hColFirstPtr, -1, ldpcN * sizeof(int));
	memset(hColNextPtr, -1, nonZeros * sizeof(int));
	//create a  adjacency links for RowPtr;
	hRowFirstPtr = (int *) malloc(ldpcM * sizeof(int));
	hRowNextPtr = (int *) malloc(nonZeros * sizeof(int));
	memset(hRowFirstPtr, -1, ldpcM * sizeof(int));
	memset(hRowNextPtr, -1, nonZeros * sizeof(int));

	hCols = (int*) malloc(nonZeros * sizeof(int));
	hRows = (int*) malloc(nonZeros * sizeof(int));

	hRowRange = (int*) malloc((ldpcM + 1) * sizeof(int));
	//init the hColPtr and hRowPtr;value is the offset in nonZeros,r0,r1 and so on;
	int offset = 0;
	for (int k = 0; k < checkMatrix.outerSize(); ++k) {
		hRowRange[k] = offset;
		for (SparseMatrix<DataType, RowMajor>::InnerIterator it(checkMatrix, k);
				it; ++it) {
			//set thre adjacency links
			int row = it.row();
			int col = it.col();
			//set the hRow and hCol
			hRows[offset] = row;
			hCols[offset] = col;
			//init the hRowPtr
			if (hRowFirstPtr[row] == -1) {
				hRowFirstPtr[row] = offset;
			} else {
				int next;
				for (next = hRowFirstPtr[row]; hRowNextPtr[next] != -1; next =
						hRowNextPtr[next]) {
				}
				hRowNextPtr[next] = offset;
			}
			//init the hColPtr
			if (hColFirstPtr[col] == -1) {
				hColFirstPtr[col] = offset;
			} else {
				int next;
				for (next = hColFirstPtr[col]; hColNextPtr[next] != -1; next =
						hColNextPtr[next]) {
				}
				hColNextPtr[next] = offset;
			}
			++offset;
		}
	}

	hRowRange[ldpcM] = offset;

	nonZeros = checkMatrix.nonZeros();
	//create platform
	std::vector<cl::Platform> platformList;
	cl::Platform::get(&platformList);
	//create context
	cl_context_properties cprops[] = { CL_CONTEXT_PLATFORM,
			(cl_context_properties) (platformList[0])(), 0 };

	context = cl::Context(CL_DEVICE_TYPE_GPU, cprops, NULL, NULL, &errNum);
	//create commandqueue
	std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
	queue = cl::CommandQueue(context, devices[0], 0, &errNum);
	//create program
	char * kernelSourceCode = load_program_source("./decodeCL.c");
	cl::Program::Sources sources(1, std::make_pair(kernelSourceCode, 0));
	program = cl::Program(context, sources, &errNum);
	free(kernelSourceCode);
	//program.build(devices);
	errNum = clBuildProgram(program(), 0, NULL, NULL, NULL, NULL);
	if (errNum != CL_SUCCESS) {
// Determine the reason for the error
		char buildLog[16384];
		clGetProgramBuildInfo(program(), devices[0](), CL_PROGRAM_BUILD_LOG,
				sizeof(buildLog), buildLog, NULL);

		printf("Error in kernel: \n");
		printf("%s\n", buildLog);
		//std::cerr << buildLog;
		clReleaseProgram(program());
		exit(0);
	}
	//create memery
	memHRow = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			nonZeros * sizeof(int), hRows);
	memHCol = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			nonZeros * sizeof(int), hCols);
	memHRowFirstPtr = cl::Buffer(context,
	CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ldpcM * sizeof(int), hRowFirstPtr);
	memHRowNextPtr = cl::Buffer(context,
	CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nonZeros * sizeof(int),
			hRowNextPtr);
	memHColFirstPtr = cl::Buffer(context,
	CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ldpcN * sizeof(int), hColFirstPtr);
	memHColNextPtr = cl::Buffer(context,
	CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nonZeros * sizeof(int),
			hColNextPtr);
	memCodes = cl::Buffer(context, CL_MEM_READ_ONLY,
			batchSize * ldpcN * sizeof(float), NULL);
	flagsBool = (bool*) malloc(batchSize * sizeof(bool));
	memFlagsBool = cl::Buffer(context, CL_MEM_READ_WRITE,
			batchSize * sizeof(bool),
			NULL, &errNum);
	// the src buffer
	srcBool = (bool*) malloc(batchSize * ldpcN * sizeof(bool));
	memSrcBool = cl::Buffer(context, CL_MEM_READ_WRITE,
			batchSize * ldpcN * sizeof(bool), NULL, &errNum);
	memSrcCode = cl::Buffer(context, CL_MEM_READ_WRITE,
			batchSize * ldpcK / 8 * sizeof(char), NULL, &errNum);

	memIsDones = cl::Buffer(context, CL_MEM_READ_WRITE,
			batchSize * sizeof(bool), NULL, &errNum);
	kerToChar = cl::Kernel(program, "toChar", &errNum);
	kerToChar.setArg(0, memSrcBool);
	kerToChar.setArg(1, memSrcCode);
	kerToChar.setArg(2, sizeof(int), &ldpcN);
	kerToChar.setArg(3, sizeof(int), &ldpcK);

	kerCheckResult = cl::Kernel(program, "checkResult", &errNum);
	kerCheckResult.setArg(0, memSrcBool);
	kerCheckResult.setArg(1, memHCol);
	kerCheckResult.setArg(2, memHRowFirstPtr);
	kerCheckResult.setArg(3, memHRowNextPtr);
	kerCheckResult.setArg(4, sizeof(int), &ldpcM);
	kerCheckResult.setArg(5, sizeof(int), &ldpcN);
	kerCheckResult.setArg(6, sizeof(int), &nonZeros);
	kerCheckResult.setArg(7, memFlagsBool);
	kerCheckResult.setArg(8, memIsDones);

	kerCheckDones = cl::Kernel(program, "checkDones", &errNum);
	kerCheckDones.setArg(0, memFlagsBool);
	kerCheckDones.setArg(1, memIsDones);
	return 0;
}
int Coder::addDecodeType(enum decodeType deType) {

	char zChar;
	char seedRowLengthChar;
	char maxColWtChar;
	int hSeedSize;
	char * hSeedChar;

	switch (deType) {
	case DecodeSP:
		isDecodeSP = true;
		//for decode by sum product algorithm------------------
		memR0 = cl::Buffer(context, CL_MEM_READ_WRITE,
				batchSize * nonZeros * sizeof(float), NULL);
		memR1 = cl::Buffer(context, CL_MEM_READ_WRITE,
				batchSize * nonZeros * sizeof(float), NULL);
		memQ0 = cl::Buffer(context, CL_MEM_READ_WRITE,
				batchSize * nonZeros * sizeof(float), NULL);
		memQ1 = cl::Buffer(context, CL_MEM_READ_WRITE,
				batchSize * nonZeros * sizeof(float), NULL);

		memPriorP0 = cl::Buffer(context, CL_MEM_READ_WRITE,
				batchSize * ldpcN * sizeof(float), NULL);
		memPriorP1 = cl::Buffer(context, CL_MEM_READ_WRITE,
				batchSize * ldpcN * sizeof(float), NULL);

		//create kernel
		kerDecodeInit = cl::Kernel(program, "decodeInit", &errNum);
		kerRefreshR = cl::Kernel(program, "refreshR", &errNum);
		kerRefreshQ = cl::Kernel(program, "refreshQ", &errNum);
		kerHardDecision = cl::Kernel(program, "hardDecision", &errNum);
		//set kernel argument
		kerDecodeInit.setArg(0, memHCol);
		kerDecodeInit.setArg(1, memCodes);
		kerDecodeInit.setArg(2, memQ0);
		kerDecodeInit.setArg(3, memQ1);
		kerDecodeInit.setArg(4, memPriorP0);
		kerDecodeInit.setArg(5, memPriorP1);
		kerDecodeInit.setArg(6, sizeof(int), &ldpcN);
		kerDecodeInit.setArg(7, sizeof(int), &nonZeros);
		kerDecodeInit.setArg(8, memIsDones);

		kerRefreshR.setArg(0, memHRow);
		kerRefreshR.setArg(1, memQ0);
		kerRefreshR.setArg(2, memQ1);
		kerRefreshR.setArg(3, memR0);
		kerRefreshR.setArg(4, memR1);
		kerRefreshR.setArg(5, memHRowFirstPtr);
		kerRefreshR.setArg(6, memHRowNextPtr);
		kerRefreshR.setArg(7, sizeof(int), &ldpcM);
		kerRefreshR.setArg(8, sizeof(int), &nonZeros);
		kerRefreshR.setArg(9, memIsDones);

		kerRefreshQ.setArg(0, memHCol);
		kerRefreshQ.setArg(1, memQ0);
		kerRefreshQ.setArg(2, memQ1);
		kerRefreshQ.setArg(3, memR0);
		kerRefreshQ.setArg(4, memR1);
		kerRefreshQ.setArg(5, memPriorP0);
		kerRefreshQ.setArg(6, memPriorP1);
		kerRefreshQ.setArg(7, memHColFirstPtr);
		kerRefreshQ.setArg(8, memHColNextPtr);
		kerRefreshQ.setArg(9, sizeof(int), &ldpcN);
		kerRefreshQ.setArg(10, sizeof(int), &nonZeros);
		kerRefreshQ.setArg(11, memFlagsBool);
		kerRefreshQ.setArg(12, memIsDones);

		kerHardDecision.setArg(0, memSrcBool);
		kerHardDecision.setArg(1, memR0);
		kerHardDecision.setArg(2, memR1);
		kerHardDecision.setArg(3, memPriorP0);
		kerHardDecision.setArg(4, memPriorP1);
		kerHardDecision.setArg(5, memHColFirstPtr);
		kerHardDecision.setArg(6, memHColNextPtr);
		kerHardDecision.setArg(7, sizeof(int), &ldpcN);
		kerHardDecision.setArg(8, sizeof(int), &nonZeros);
		kerHardDecision.setArg(9, memFlagsBool);
		kerHardDecision.setArg(10, memIsDones);

		break;
	case DecodeMS:
		isDecodeMS = true;
		//for decode by sum product algorithm------------------
		memLR = cl::Buffer(context, CL_MEM_READ_WRITE,
				batchSize * nonZeros * sizeof(float), NULL);
		memLQ = cl::Buffer(context, CL_MEM_READ_WRITE,
				batchSize * nonZeros * sizeof(float), NULL);
		memLPostP = cl::Buffer(context, CL_MEM_READ_WRITE,
				batchSize * ldpcN * sizeof(float), NULL);
		// the flags
		kerDecodeInit = cl::Kernel(program, "decodeInitMS", &errNum);
		kerRefreshR = cl::Kernel(program, "refreshRMS", &errNum);
		kerRefreshQ = cl::Kernel(program, "refreshQMS", &errNum);
		kerRefreshPostP = cl::Kernel(program, "refreshPostPMS", &errNum);

		kerDecodeInit.setArg(0, memHCol);
		kerDecodeInit.setArg(1, memCodes);
		kerDecodeInit.setArg(2, memLQ);
		kerDecodeInit.setArg(3, sizeof(int), &ldpcN);
		kerDecodeInit.setArg(4, sizeof(int), &nonZeros);
		kerDecodeInit.setArg(5, memIsDones);

		kerRefreshR.setArg(0, memHRow);
		kerRefreshR.setArg(1, memLQ);
		kerRefreshR.setArg(2, memLR);
		kerRefreshR.setArg(3, memHRowFirstPtr);
		kerRefreshR.setArg(4, memHRowNextPtr);
		kerRefreshR.setArg(5, sizeof(int), &nonZeros);
		kerRefreshR.setArg(6, memIsDones);

		kerRefreshPostP.setArg(0, memSrcBool);
		kerRefreshPostP.setArg(1, memLR);
		kerRefreshPostP.setArg(2, memCodes);
		kerRefreshPostP.setArg(3, memLPostP);
		kerRefreshPostP.setArg(4, memHColFirstPtr);
		kerRefreshPostP.setArg(5, memHColNextPtr);
		kerRefreshPostP.setArg(6, sizeof(int), &ldpcN);
		kerRefreshPostP.setArg(7, sizeof(int), &nonZeros);
		kerRefreshPostP.setArg(8, memFlagsBool);
		kerRefreshPostP.setArg(9, memIsDones);

		kerRefreshQ.setArg(0, memHCol);
		kerRefreshQ.setArg(1, memLQ);
		kerRefreshQ.setArg(2, memLR);
		kerRefreshQ.setArg(3, memLPostP);
		kerRefreshQ.setArg(4, sizeof(int), &ldpcN);
		kerRefreshQ.setArg(5, sizeof(int), &nonZeros);
		kerRefreshQ.setArg(6, memFlagsBool);
		kerRefreshQ.setArg(7, memIsDones);

		break;
	case DecodeCPU:
		break;
	case DecodeTDMP:
		blockHeavy = ldpcN;

		memLR = cl::Buffer(context, CL_MEM_READ_WRITE,
				batchSize * nonZeros * sizeof(float), NULL);
		memLQ = cl::Buffer(context, CL_MEM_READ_WRITE,
				batchSize * blockHeavy * sizeof(float), NULL);
		memLPostP = cl::Buffer(context, CL_MEM_READ_WRITE,
				batchSize * ldpcN * sizeof(float), NULL);
		// the flags
		kerDecodeInit = cl::Kernel(program, "decodeInitTDMP", &errNum);
		kerRefreshR = cl::Kernel(program, "refreshRTDMP", &errNum);
		kerRefreshQ = cl::Kernel(program, "refreshQTDMP", &errNum);
		kerRefreshPostP = cl::Kernel(program, "refreshPostPTDMP", &errNum);
		kerHardDecision = cl::Kernel(program, "hardDecisionTDMP", &errNum);

		kerDecodeInit.setArg(0, memHCol);
		kerDecodeInit.setArg(1, memCodes);
		kerDecodeInit.setArg(2, memLQ);
		kerDecodeInit.setArg(3, sizeof(int), &ldpcN);
		kerDecodeInit.setArg(4, sizeof(int), &nonZeros);
		kerDecodeInit.setArg(5, memIsDones);
		kerDecodeInit.setArg(6, memLPostP);
		kerDecodeInit.setArg(7, sizeof(int), &blockHeavy);
		kerDecodeInit.setArg(8, memLR);

		kerRefreshR.setArg(0, memHRow);
		kerRefreshR.setArg(1, memLQ);
		kerRefreshR.setArg(2, memLR);
		kerRefreshR.setArg(3, memHRowFirstPtr);
		kerRefreshR.setArg(4, memHRowNextPtr);
		kerRefreshR.setArg(5, sizeof(int), &nonZeros);
		kerRefreshR.setArg(6, memIsDones);
		kerRefreshR.setArg(7, sizeof(int), &blockHeavy);
		//kerRefreshR.setArg(8, sizeof(int,&offset);

		kerRefreshPostP.setArg(0, memLR);
		kerRefreshPostP.setArg(1, memLQ);
		kerRefreshPostP.setArg(2, memLPostP);
		kerRefreshPostP.setArg(3, memHCol);
		kerRefreshPostP.setArg(4, sizeof(int), &ldpcN);
		kerRefreshPostP.setArg(5, sizeof(int), &nonZeros);
		kerRefreshPostP.setArg(6, memIsDones);
		kerRefreshPostP.setArg(7, sizeof(int), &blockHeavy);
		//kerRefreshPostP.setArg(8, sizeof(int),&offset);

		kerRefreshQ.setArg(0, memHCol);
		kerRefreshQ.setArg(1, memLQ);
		kerRefreshQ.setArg(2, memLR);
		kerRefreshQ.setArg(3, memLPostP);
		kerRefreshQ.setArg(4, sizeof(int), &ldpcN);
		kerRefreshQ.setArg(5, sizeof(int), &nonZeros);
		kerRefreshQ.setArg(6, memFlagsBool);
		kerRefreshQ.setArg(7, memIsDones);
		kerRefreshQ.setArg(8, sizeof(int), &blockHeavy);
		//kerRefreshQ.setArg(9, sizeof(int),&offset);

		kerHardDecision.setArg(0, memLPostP);
		kerHardDecision.setArg(1, memSrcBool);
		kerHardDecision.setArg(2, memFlagsBool);
		kerHardDecision.setArg(3, sizeof(int), &ldpcN);
		kerHardDecision.setArg(4, memIsDones);
		break;
	case DecodeTDMPCL:
		zChar = (char) z;
		seedRowLengthChar = (char) (ldpcM / z);
		maxColWtChar = (char) maxColWt;
		hSeedSize = ldpcM / z * ldpcN / z;
		hSeedChar = (char*) malloc(sizeof(char) * hSeedSize);
		for (int i = 0; i < hSeedSize; ++i)
			hSeedChar[i] = hSeed[i];
		//strncpy(hSeedChar, hSeed, hSeedSize); ??why
		memHSeed = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				hSeedSize * sizeof(char), hSeedChar);
		kerDecodeOnceTDMP = cl::Kernel(program, "decodeOnceTDMP", &errNum);
		kerDecodeOnceTDMP.setArg(0, memCodes);
		kerDecodeOnceTDMP.setArg(1, memSrcCode);
		kerDecodeOnceTDMP.setArg(2, sizeof(char), &zChar);
		kerDecodeOnceTDMP.setArg(3, sizeof(char), &seedRowLengthChar);
		kerDecodeOnceTDMP.setArg(4, memHSeed);
		kerDecodeOnceTDMP.setArg(5, sizeof(float) * ldpcN, NULL);
		kerDecodeOnceTDMP.setArg(6, sizeof(bool) * ldpcN, NULL);
		kerDecodeOnceTDMP.setArg(7, sizeof(bool), NULL);

		break;
	case DecodeMSCL:

		zChar = (char) z;
		seedRowLengthChar = (char) (ldpcM / z);
		maxColWtChar = (char) maxColWt;
		hSeedSize = ldpcM / z * ldpcN / z;
		hSeedChar = (char*) malloc(sizeof(char) * hSeedSize);

		for (int i = 0; i < hSeedSize; ++i)
			hSeedChar[i] = hSeed[i];
		//strncpy(hSeedChar, hSeed, hSeedSize); ??why
		memHSeed = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				hSeedSize * sizeof(char), hSeedChar);
		kerDecodeOnceMS = cl::Kernel(program, "decodeOnceMS", &errNum);
		kerDecodeOnceMS.setArg(0, memCodes);
		kerDecodeOnceMS.setArg(1, memSrcCode);
		kerDecodeOnceMS.setArg(2, sizeof(char), &zChar);
		kerDecodeOnceMS.setArg(3, sizeof(char), &seedRowLengthChar);
		kerDecodeOnceMS.setArg(4, memHSeed);
		kerDecodeOnceMS.setArg(5, sizeof(float) * ldpcN, NULL);
		kerDecodeOnceMS.setArg(6, sizeof(float) * ldpcN * ldpcM / z, NULL);
		kerDecodeOnceMS.setArg(7, sizeof(bool) * ldpcN, NULL);
		kerDecodeOnceMS.setArg(8, sizeof(bool), NULL);

		break;
	}
	return LDPC_SUCCESS;
}

int Coder::encode(char * srcCode, char * priorCode, int srcLength) {
	int codeSize = getCodeSize(srcLength);
	for (int offset = 0;; offset += 1) {
		if ((offset + 1) * ldpcK / 8 < srcLength) { //not the last;
			int srcL = ldpcK / 8;
			encodeOnce(&srcCode[offset * ldpcK / 8],
					&priorCode[offset * ldpcN / 8], srcL);
		} else { // the last
			int srcL = srcLength - offset * ldpcK / 8;
			encodeOnce(&srcCode[offset * ldpcK / 8],
					&priorCode[offset * ldpcN / 8], srcL);
			break;
		}
	}
	return LDPC_SUCCESS;
}

int Coder::decode(float * postCode, char * srcCode, int srcLength,
		enum decodeType deType) {
	if (deType == DecodeCPU) {
		return decodeCPU(postCode, srcCode, srcLength);
	}
	int codeSize = getCodeSize(srcLength);
	for (int offset = 0;; offset += batchSize) {
		if (offset + batchSize < codeSize) { //not the last
			int bat = batchSize;
			int srcL = bat * ldpcK / 8;
			if (deType == DecodeMS)
				decodeOnceMS(&postCode[offset * ldpcN],
						&srcCode[offset * ldpcK / 8], bat * ldpcN, srcL);
			else if (deType == DecodeSP)
				decodeOnceSP(&postCode[offset * ldpcN],
						&srcCode[offset * ldpcK / 8], bat * ldpcN, srcL);
			else if (deType == DecodeTDMP)
				decodeOnceTDMP(&postCode[offset * ldpcN],
						&srcCode[offset * ldpcK / 8], bat * ldpcN, srcL);
			else if (deType == DecodeTDMPCL)
				decodeOnceTDMPCL(&postCode[offset * ldpcN],
						&srcCode[offset * ldpcK / 8], bat * ldpcN, srcL);
			else if (deType == DecodeMSCL)
				decodeOnceMSCL(&postCode[offset * ldpcN],
						&srcCode[offset * ldpcK / 8], bat * ldpcN, srcL);
		} else { //is the last
			int bat = codeSize - offset;
			int srcL = srcLength - offset * ldpcK / 8;
			if (deType == DecodeMS)
				decodeOnceMS(&postCode[offset * ldpcN],
						&srcCode[offset * ldpcK / 8], bat * ldpcN, srcL);
			else if (deType == DecodeSP)
				decodeOnceSP(&postCode[offset * ldpcN],
						&srcCode[offset * ldpcK / 8], bat * ldpcN, srcL);
			else if (deType == DecodeTDMP)
				decodeOnceTDMP(&postCode[offset * ldpcN],
						&srcCode[offset * ldpcK / 8], bat * ldpcN, srcL);
			else if (deType == DecodeTDMPCL)
				decodeOnceTDMPCL(&postCode[offset * ldpcN],
						&srcCode[offset * ldpcK / 8], bat * ldpcN, srcL);
			else if (deType == DecodeMSCL)
				decodeOnceMSCL(&postCode[offset * ldpcN],
						&srcCode[offset * ldpcK / 8], bat * ldpcN, srcL);
			break;
		}
	}
	return LDPC_SUCCESS;
}

int Coder::getPriorCodeLength(int srcLength) {
	return (srcLength + (ldpcK / 8) - 1) / (ldpcK / 8) * (ldpcN / 8);
}

int Coder::getPostCodeLength(int srcLength) {
	return (srcLength + (ldpcK / 8) - 1) / (ldpcK / 8) * ldpcN;
}

int Coder::getCodeSize(int srcLength) {
	return (srcLength + (ldpcK / 8) - 1) / (ldpcK / 8);

}

int Coder::encodeOnce(char * srcCode, char * priorCode, int srcLength) {
	using namespace Eigen;
	smK.setZero();
	smP1.setZero();
	smP2.setZero();
//init smK;
	for (int charOffset = 0; charOffset < ldpcK / 8; ++charOffset) {
		char tmp;
		if (charOffset < srcLength) {
			tmp = srcCode[charOffset];
			for (int bitOffset = 0; bitOffset < 8; ++bitOffset) {
				char flag = 1 << bitOffset;
				if (flag & tmp) {
					smK.insert(8 * charOffset + bitOffset, 0) = 1;
				}
			}
		}
	}
//calculate the smP1(z,1) and smP2(ldpcM-z,1)
	smP1 = smTmp * smK;
	using namespace std;
	binarySM(smP1);
	smP1.makeCompressed();
	smP2 = smInvT * (smA * smK + smB * smP1);
	binarySM(smP2);
	smP2.makeCompressed();

//change from sm to char ptr;
	strncpy(priorCode, srcCode, srcLength);
	memset(priorCode + srcLength, 0, ldpcN / 8 - srcLength);

	for (SparseMatrix<DataType>::InnerIterator it(smP1, 0); it; ++it) {
		if (it.value()) {
			int offset = it.row() + ldpcK;   // row index
			int charOffset = offset / 8;
			int bitOffset = offset % 8;
			priorCode[charOffset] |= (1 << bitOffset);
		}
	}

	for (SparseMatrix<DataType>::InnerIterator it(smP2, 0); it; ++it) {
		if (it.value()) {
			int offset = it.row() + ldpcK + z;   // row index
			int charOffset = offset / 8;
			int bitOffset = offset % 8;
			priorCode[charOffset] |= (1 << bitOffset);
		}
	}
	return LDPC_SUCCESS;
}

int Coder::decodeCPU(float * postCode, char * srcCode, int srcLength) {
	memset(srcCode, 0, srcLength);
	int codeSize = getCodeSize(srcLength);
	bool* lQA = (bool*) malloc(sizeof(bool) * nonZeros);
	float* lQB = (float*) malloc(sizeof(float) * nonZeros);
	float* lR = (float*) malloc(sizeof(float) * nonZeros);
	float* lPriorP = (float*) malloc(sizeof(float) * ldpcN);
	float* lPostP = (float*) malloc(sizeof(float) * ldpcN);
	bool * src = (bool*) malloc(sizeof(bool) * ldpcN);
	int time;
	for (int batch = 0; batch < codeSize; ++batch) {
		//decodeInitMS
		time = 0;
		for (int nodeInd = 0; nodeInd < nonZeros; ++nodeInd) {
			int hCol = hCols[nodeInd];
			float code = postCode[batch * ldpcN + hCol];
			lQA[nodeInd] = (code < 0);
			lQB[nodeInd] = fabs(code);
		}
		while (1) {
			//refreshRMS
			for (int nodeInd = 0; nodeInd < nonZeros; ++nodeInd) {
				int hRow = hRows[nodeInd];
				bool a = 0;
				float b = 1000;
				for (int ptr = hRowFirstPtr[hRow]; ptr != -1; ptr =
						hRowNextPtr[ptr]) {
					if (nodeInd == ptr)
						continue;
					if (lQA[ptr])
						a = a ^ 1;
					b = fmin(b, lQB[ptr]);
				}
				if (a)
					lR[nodeInd] = -b;
				else
					lR[nodeInd] = b;
			}
			//refreshPostPMS
			for (int nodeInd = 0; nodeInd < ldpcN; ++nodeInd) {
				float tmp = postCode[batch * ldpcN + nodeInd];
				for (int ptr = hColFirstPtr[nodeInd]; ptr != -1; ptr =
						hColNextPtr[ptr]) {
					tmp += lR[ptr];
				}
				if (tmp > 0) {
					src[nodeInd] = 0;
				} else {
					src[nodeInd] = 1;   //error
				}
				lPostP[nodeInd] = tmp;
			}
			//checkResult
			bool flag = 0;
			for (int nodeInd = 0; nodeInd < ldpcM; ++nodeInd) {
				bool result = 0;
				for (int ptr = hRowFirstPtr[nodeInd]; ptr != -1; ptr =
						hRowNextPtr[ptr]) {
					//ptr is the node location;
					if (src[hCols[ptr]])
						result ^= 1;
				}
				if (result) {
					flag = 1;
					break;
				}
			}
			++time;
			if (flag == 0)
				break;
			if (time == times)
				break;
			//refreshQ
			for (int nodeInd = 0; nodeInd < nonZeros; ++nodeInd) {
				int hCol = hCols[nodeInd];
				float lQ = lPostP[hCol] - lR[nodeInd];
				lQA[nodeInd] = (lQ < 0);
				lQB[nodeInd] = fabs(lQ);
			}
		}   //while done
			//output
		for (int tmp = 0; tmp < ldpcK; ++tmp) {
			if (src[tmp]) {
				int offset = batch * ldpcK + tmp;
				int charOffset = offset / 8;
				if (charOffset <= srcLength) {
					int bitOffset = offset % 8;
					srcCode[charOffset] |= (1 << bitOffset);
				}
			}
		}

	}
	free(lQA);
	free(lQB);
	free(lPriorP);
	free(lPostP);
	free(src);
	free(lR);
	return 0;
}

int Coder::decodeOnceMS(float * postCode, char * srcCode, int postCodeLength,
		int srcLength) {
	using namespace std;
	int time = 0;
	cl::Event eventDecodeInit, eventRefreshR, eventRefreshQ, eventRefreshPostP,
			eventCheckResult, eventToChar, eventWriteBuffer;
	int batchSizeOnce = postCodeLength / ldpcN;
	clock_t start;

	start = clock();
	errNum = queue.enqueueWriteBuffer(memCodes, CL_TRUE, 0,
			postCodeLength * sizeof(float), postCode, NULL, &eventWriteBuffer);
	queue.enqueueNDRangeKernel(kerDecodeInit, cl::NullRange,
			cl::NDRange(batchSizeOnce, nonZeros), cl::NullRange,
			new std::vector<cl::Event>(1, eventWriteBuffer), &eventRefreshQ); //actually is eventDecodeInit;

	start = clock();
	while (1) {
		queue.enqueueNDRangeKernel(kerRefreshR, cl::NullRange,
				cl::NDRange(batchSizeOnce, nonZeros), cl::NullRange,
				new std::vector<cl::Event>(1, eventRefreshQ), &eventRefreshR);

		queue.enqueueNDRangeKernel(kerRefreshPostP, cl::NullRange,
				cl::NDRange(batchSizeOnce, ldpcN + 1), cl::NullRange,
				new std::vector<cl::Event>(1, eventRefreshR),
				&eventRefreshPostP);

		queue.enqueueNDRangeKernel(kerCheckResult, cl::NullRange,
				cl::NDRange(batchSizeOnce, ldpcM), cl::NullRange,
				new std::vector<cl::Event>(1, eventRefreshPostP),
				&eventCheckResult);

		errNum = queue.enqueueReadBuffer(memFlagsBool, CL_TRUE, 0,
				batchSizeOnce * sizeof(bool), flagsBool,
				new std::vector<cl::Event>(1, eventCheckResult),
				NULL);
		queue.finish();

		++time;
		int sumFlag = 0;
		for (int i = 0; i < batchSizeOnce; ++i)
			if (flagsBool[i])
				++sumFlag;

		if (sumFlag == 0)
			break;
		if (time == times)
			break;
		queue.enqueueNDRangeKernel(kerRefreshQ, cl::NullRange,
				cl::NDRange(batchSizeOnce, nonZeros), cl::NullRange,
				NULL, &eventRefreshQ);
	}
	cout << "Time=" << time << endl;
	errNum = queue.enqueueNDRangeKernel(kerToChar, cl::NullRange,
			cl::NDRange(batchSizeOnce, ldpcK / 8), cl::NullRange, NULL,
			&eventToChar);
	errNum = queue.enqueueReadBuffer(memSrcCode, CL_TRUE, 0,
			srcLength * sizeof(char), srcCode,
			new std::vector<cl::Event>(1, eventToChar), NULL);
	queue.finish();

	return LDPC_SUCCESS;
}

int Coder::decodeOnceTDMPCL(float * postCode, char * srcCode,
		int postCodeLength, int srcLength) {
	int batchSizeOnce = postCodeLength / ldpcN;
	cl::Event eventWriteBuffer, eventReadBuffer, eventDecode;
	errNum = queue.enqueueWriteBuffer(memCodes, CL_TRUE, 0,
			postCodeLength * sizeof(float), postCode, NULL, &eventWriteBuffer);
	queue.finish();
	errNum = queue.enqueueNDRangeKernel(kerDecodeOnceTDMP, cl::NullRange,
			cl::NDRange(batchSizeOnce * z), cl::NDRange(z),
			new std::vector<cl::Event>(1, eventWriteBuffer), &eventDecode);
	queue.finish();
	errNum = queue.enqueueReadBuffer(memSrcCode, CL_TRUE, 0,
			srcLength * sizeof(char), srcCode,
			new std::vector<cl::Event>(1, eventDecode), NULL);
	queue.finish();

	return 0;

}

int Coder::decodeOnceMSCL(float * postCode, char * srcCode,
		int postCodeLength, int srcLength) {
	int batchSizeOnce = postCodeLength / ldpcN;
	cl::Event eventWriteBuffer, eventReadBuffer, eventDecode;
	errNum = queue.enqueueWriteBuffer(memCodes, CL_TRUE, 0,
			postCodeLength * sizeof(float), postCode, NULL, &eventWriteBuffer);
	queue.finish();
	errNum = queue.enqueueNDRangeKernel(kerDecodeOnceMS, cl::NullRange,
			cl::NDRange(batchSizeOnce * z,ldpcM/z), cl::NDRange(z,ldpcM/z),
			new std::vector<cl::Event>(1, eventWriteBuffer), &eventDecode);
	queue.finish();
	errNum = queue.enqueueReadBuffer(memSrcCode, CL_TRUE, 0,
			srcLength * sizeof(char), srcCode,
			new std::vector<cl::Event>(1, eventDecode), NULL);
	queue.finish();

	return 0;

}
int Coder::decodeOnceTDMP(float * postCode, char * srcCode, int postCodeLength,
		int srcLength) {
	using namespace std;
	int time = 0;
	cl::Event eventDecodeInit, eventRefreshR, eventRefreshQ, eventRefreshPostP,
			eventCheckResult, eventToChar, eventHardDecision, eventWriteBuffer;
	int batchSizeOnce = postCodeLength / ldpcN;

	errNum = queue.enqueueWriteBuffer(memCodes, CL_TRUE, 0,
			postCodeLength * sizeof(float), postCode, NULL, &eventWriteBuffer);

	queue.enqueueNDRangeKernel(kerDecodeInit, cl::NullRange,
			cl::NDRange(batchSizeOnce, ldpcN + 1), cl::NullRange,
			new std::vector<cl::Event>(1, eventWriteBuffer), &eventRefreshQ); //actually is eventDecodeInit;

	int offset = 0;
	int blockRow = 0;
	int threadNum;
	threadNum = hRowRange[blockRow + z] - hRowRange[blockRow];
	while (1) {

		kerRefreshR.setArg(8, sizeof(int), &offset);
		queue.enqueueNDRangeKernel(kerRefreshR, cl::NullRange,
				cl::NDRange(batchSizeOnce, threadNum), cl::NullRange,
				new std::vector<cl::Event>(1, eventRefreshQ), &eventRefreshR);

		kerRefreshPostP.setArg(8, sizeof(int), &offset);
		queue.enqueueNDRangeKernel(kerRefreshPostP, cl::NullRange,
				cl::NDRange(batchSizeOnce, threadNum), cl::NullRange,
				new std::vector<cl::Event>(1, eventRefreshR),
				&eventRefreshPostP);

		++blockRow;
		offset += threadNum;
		if (blockRow == ldpcM / z) {
			queue.enqueueNDRangeKernel(kerHardDecision, cl::NullRange,
					cl::NDRange(batchSizeOnce, ldpcN + 1), cl::NullRange,
					new std::vector<cl::Event>(1, eventRefreshPostP),
					&eventHardDecision);

			queue.enqueueNDRangeKernel(kerCheckResult, cl::NullRange,
					cl::NDRange(batchSizeOnce, ldpcM), cl::NullRange,
					new std::vector<cl::Event>(1, eventHardDecision),
					&eventCheckResult);

			queue.enqueueNDRangeKernel(kerCheckDones, cl::NullRange,
					cl::NDRange(batchSizeOnce, 1), cl::NullRange,
					new std::vector<cl::Event>(1, eventCheckResult),
					NULL);

			errNum = queue.enqueueReadBuffer(memFlagsBool, CL_TRUE, 0,
					batchSizeOnce * sizeof(bool), flagsBool,
					new std::vector<cl::Event>(1, eventCheckResult),
					NULL);
			queue.finish();

			++time;
			int sumFlag = 0;
			for (int i = 0; i < batchSizeOnce; ++i)
				if (flagsBool[i])
					++sumFlag;

			if (sumFlag == 0)
				break;
			if (time == times)
				break;
			blockRow = 0;
			offset = 0;
		}
		threadNum = hRowRange[blockRow + z] - hRowRange[blockRow];
		kerRefreshQ.setArg(9, sizeof(int), &offset);
		queue.enqueueNDRangeKernel(kerRefreshQ, cl::NullRange,
				cl::NDRange(batchSizeOnce, threadNum), cl::NullRange,
				new std::vector<cl::Event>(1, eventRefreshPostP),
				&eventRefreshQ);

	}
	cout << "Time=" << time << endl;
	errNum = queue.enqueueNDRangeKernel(kerToChar, cl::NullRange,
			cl::NDRange(batchSizeOnce, ldpcK / 8), cl::NullRange, NULL,
			&eventToChar);
	errNum = queue.enqueueReadBuffer(memSrcCode, CL_TRUE, 0,
			srcLength * sizeof(char), srcCode,
			new std::vector<cl::Event>(1, eventToChar), NULL);
	queue.finish();

	return LDPC_SUCCESS;
}
int Coder::decodeOnceSP(float * postCode, char * srcCode, int postCodeLength,
		int srcLength) {
	using namespace std;
	int time = 0;
	int sumFlag = 1;
	clock_t start;
	cl::Event eventDecodeInit, eventRefreshR, eventRefreshQ, eventHardDecision,
			eventCheckResult, eventToChar;
	int batchSizeOnce = postCodeLength / ldpcN;

	start = clock();
	errNum = queue.enqueueWriteBuffer(memCodes, CL_TRUE, 0,
			postCodeLength * sizeof(float), postCode, NULL, NULL);
	stepTime[0] += (double) (clock() - start) / CLOCKS_PER_SEC;

	start = clock();
	queue.enqueueNDRangeKernel(kerDecodeInit, cl::NullRange,
			cl::NDRange(batchSizeOnce, nonZeros), cl::NullRange, NULL,
			&eventRefreshQ);
	queue.finish();
	stepTime[1] += (double) (clock() - start) / CLOCKS_PER_SEC;

	start = clock();
	while (1) {
		queue.enqueueNDRangeKernel(kerRefreshR, cl::NullRange,
				cl::NDRange(batchSizeOnce, nonZeros), cl::NullRange,
				new std::vector<cl::Event>(1, eventRefreshQ), &eventRefreshR);
		queue.finish();
		stepTime[2] += (double) (clock() - start) / CLOCKS_PER_SEC;

		start = clock();
		queue.enqueueNDRangeKernel(kerHardDecision, cl::NullRange,
				cl::NDRange(batchSizeOnce, nonZeros), cl::NullRange,
				new std::vector<cl::Event>(1, eventRefreshR),
				&eventHardDecision);
		queue.finish();
		stepTime[3] += (double) (clock() - start) / CLOCKS_PER_SEC;

		start = clock();
		queue.enqueueNDRangeKernel(kerCheckResult, cl::NullRange,
				cl::NDRange(batchSizeOnce, nonZeros), cl::NullRange,
				new std::vector<cl::Event>(1, eventHardDecision),
				&eventCheckResult);
		queue.finish();
		stepTime[4] += (double) (clock() - start) / CLOCKS_PER_SEC;

		start = clock();
		errNum = queue.enqueueReadBuffer(memFlagsBool, CL_TRUE, 0,
				batchSizeOnce * sizeof(bool), flagsBool,
				new std::vector<cl::Event>(1, eventCheckResult), NULL);
		queue.finish();
		stepTime[5] += (double) (clock() - start) / CLOCKS_PER_SEC;

		start = clock();
		sumFlag = 0;
		for (int i = 0; i < batchSizeOnce; ++i) {
			sumFlag += flagsBool[i];
		}
		++time;
		if (sumFlag == 0)
			break;
		if (time == times)
			break;
		queue.enqueueNDRangeKernel(kerRefreshQ, cl::NullRange,
				cl::NDRange(batchSizeOnce, nonZeros), cl::NullRange,
				NULL, &eventRefreshQ);
		queue.finish();
		stepTime[6] += (double) (clock() - start) / CLOCKS_PER_SEC;

		start = clock();
	}
	cout << "Time=" << time << endl;
	errNum = queue.enqueueNDRangeKernel(kerToChar, cl::NullRange,
			cl::NDRange(batchSizeOnce, ldpcK / 8), cl::NullRange, NULL,
			&eventToChar);
	errNum = queue.enqueueReadBuffer(memSrcCode, CL_TRUE, 0,
			srcLength * sizeof(char), srcCode,
			new std::vector<cl::Event>(1, eventToChar), NULL);
	queue.finish();
	stepTime[7] += (double) (clock() - start) / CLOCKS_PER_SEC;

	return LDPC_SUCCESS;
}

int Coder::test(char* priorCode, float * postCode, int priorCodeLength,
		float rate = 0.2) {
	for (int charOffset = 0; charOffset < priorCodeLength; ++charOffset) {
		char tmp = priorCode[charOffset];
		for (int bitOffset = 0; bitOffset < 8; ++bitOffset) {
			if (tmp & (1 << bitOffset)) {	//code==1
				postCode[charOffset * 8 + bitOffset] = -1.0;
			} else {	// code ==0
				postCode[charOffset * 8 + bitOffset] = 1.0;
			}
		}
	}
	for (int charOffset = 0; charOffset < priorCodeLength * 8; ++charOffset) {
		float noise = gaussian(0, rate);
		postCode[charOffset] += noise;
	}
	return LDPC_SUCCESS;
}
char * load_program_source(const char *filename) {
	struct stat statbuf;
	FILE *fh;
	char *source;
	fh = fopen(filename, "r");
	if (fh == 0)
		return 0;
	stat(filename, &statbuf);
	source = (char *) malloc(statbuf.st_size + 1);
	fread(source, statbuf.st_size, 1, fh);
	source[statbuf.st_size] = '\0';
	return source;
}

float gaussian(float ave, float sd)  //  mean=ave,sd=Standard Deviation
		{
	float r, t, z, x;
	float s1, s2;
	const float pi = 3.1415926;
	s1 = (1.0 + rand()) / (RAND_MAX + 1.0);
	s2 = (1.0 + rand()) / (RAND_MAX + 1.0);
	r = sqrt(-2 * log(s2));
	t = 2 * pi * s1;
	z = r * cos(t);
	x = ave + z * sd;
	return x;
}

const char* openclErr2Str(cl_int error) {
#define CASE_CL_CONSTANT(NAME) case NAME: return #NAME;
// Suppose that no combinations are possible.
	switch (error) {
	CASE_CL_CONSTANT(CL_SUCCESS)
	CASE_CL_CONSTANT(CL_DEVICE_NOT_FOUND)
	CASE_CL_CONSTANT(CL_DEVICE_NOT_AVAILABLE)
	CASE_CL_CONSTANT(CL_COMPILER_NOT_AVAILABLE)
	CASE_CL_CONSTANT(CL_MEM_OBJECT_ALLOCATION_FAILURE)
	CASE_CL_CONSTANT(CL_OUT_OF_RESOURCES)
	CASE_CL_CONSTANT(CL_OUT_OF_HOST_MEMORY)
	CASE_CL_CONSTANT(CL_PROFILING_INFO_NOT_AVAILABLE)
	CASE_CL_CONSTANT(CL_MEM_COPY_OVERLAP)
	CASE_CL_CONSTANT(CL_IMAGE_FORMAT_MISMATCH)
	CASE_CL_CONSTANT(CL_IMAGE_FORMAT_NOT_SUPPORTED)
	CASE_CL_CONSTANT(CL_BUILD_PROGRAM_FAILURE)
	CASE_CL_CONSTANT(CL_MAP_FAILURE)
	CASE_CL_CONSTANT(CL_MISALIGNED_SUB_BUFFER_OFFSET)
	CASE_CL_CONSTANT(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
	CASE_CL_CONSTANT(CL_INVALID_VALUE)
	CASE_CL_CONSTANT(CL_INVALID_DEVICE_TYPE)
	CASE_CL_CONSTANT(CL_INVALID_PLATFORM)
	CASE_CL_CONSTANT(CL_INVALID_DEVICE)
	CASE_CL_CONSTANT(CL_INVALID_CONTEXT)
	CASE_CL_CONSTANT(CL_INVALID_QUEUE_PROPERTIES)
	CASE_CL_CONSTANT(CL_INVALID_COMMAND_QUEUE)
	CASE_CL_CONSTANT(CL_INVALID_HOST_PTR)
	CASE_CL_CONSTANT(CL_INVALID_MEM_OBJECT)
	CASE_CL_CONSTANT(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
	CASE_CL_CONSTANT(CL_INVALID_IMAGE_SIZE)
	CASE_CL_CONSTANT(CL_INVALID_SAMPLER)
	CASE_CL_CONSTANT(CL_INVALID_BINARY)
	CASE_CL_CONSTANT(CL_INVALID_BUILD_OPTIONS)
	CASE_CL_CONSTANT(CL_INVALID_PROGRAM)
	CASE_CL_CONSTANT(CL_INVALID_PROGRAM_EXECUTABLE)
	CASE_CL_CONSTANT(CL_INVALID_KERNEL_NAME)
	CASE_CL_CONSTANT(CL_INVALID_KERNEL_DEFINITION)
	CASE_CL_CONSTANT(CL_INVALID_KERNEL)
	CASE_CL_CONSTANT(CL_INVALID_ARG_INDEX)
	CASE_CL_CONSTANT(CL_INVALID_ARG_VALUE)
	CASE_CL_CONSTANT(CL_INVALID_ARG_SIZE)
	CASE_CL_CONSTANT(CL_INVALID_KERNEL_ARGS)
	CASE_CL_CONSTANT(CL_INVALID_WORK_DIMENSION)
	CASE_CL_CONSTANT(CL_INVALID_WORK_GROUP_SIZE)
	CASE_CL_CONSTANT(CL_INVALID_WORK_ITEM_SIZE)
	CASE_CL_CONSTANT(CL_INVALID_GLOBAL_OFFSET)
	CASE_CL_CONSTANT(CL_INVALID_EVENT_WAIT_LIST)
	CASE_CL_CONSTANT(CL_INVALID_EVENT)
	CASE_CL_CONSTANT(CL_INVALID_OPERATION)
	CASE_CL_CONSTANT(CL_INVALID_GL_OBJECT)
	CASE_CL_CONSTANT(CL_INVALID_BUFFER_SIZE)
	CASE_CL_CONSTANT(CL_INVALID_MIP_LEVEL)
	CASE_CL_CONSTANT(CL_INVALID_GLOBAL_WORK_SIZE)
	CASE_CL_CONSTANT(CL_INVALID_PROPERTY)

	default:
		return "UNKNOWN ERROR CODE";
	}

#undef CASE_CL_CONSTANT
}
