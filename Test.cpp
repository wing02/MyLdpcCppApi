/*
 * MyTest.cpp
 *
 *  Created on: Feb 15, 2016
 *      Author: wing
 */

#include<iostream>
#include<time.h>
#include"cl.hpp"
#include "MyLdpc.h"
using namespace std;

//argv[1]=srcLength,argv[2]=batchSize;
int main(int argc, char ** argv) {
	if (argc != 5)
		return 0;
	clock_t start, end;
	const int z = 48;
	//const int z = 24;
	const int ldpcN = z * 24;
	//const int ldpcK = ldpcN / 2 * 1;
	const int ldpcK = ldpcN / 6 * 5;
	const int ldpcM = ldpcN - ldpcK;
	const enum rate_type rate = rate_5_6;
	//const enum rate_type rate = rate_1_2;

	Coder coder(ldpcK, ldpcN, rate);
	srand(time(0));

	//srand(atoi(argv[5]));
	float throughPut;
	int errNum;

	int srcLength = atoi(argv[1]);
	//int srcLength = 10;
	char * srcCode = (char*) malloc(srcLength * sizeof(char));
	char * priorCode = (char*) malloc(coder.getPriorCodeLength(srcLength));
	float * postCode = (float*) malloc(
			coder.getPostCodeLength(srcLength) * sizeof(float));
	char * newSrcCode = (char*) malloc(srcLength);

	for (int i = 0; i < srcLength; i++) {
		srcCode[i] = 'a' + i % 26;
	}
	double decodeTime;
	coder.forEncoder();
	coder.forDecoder(atoi(argv[2]));
	//coder.forDecoder(1);
	start = clock();
	coder.encode(srcCode, priorCode, srcLength);
	end = clock();
	double encodeTime = (double) (end - start) / CLOCKS_PER_SEC;
	//cout << "encode time=" << encodeTime << endl;

	float snr = atof(argv[3]);
	float sd = 1 / (pow(10, snr / 20));
	cout << "sd=" << sd << endl;
	coder.test(priorCode, postCode, coder.getPriorCodeLength(srcLength), sd);

	if (!strcmp(argv[4], "SP")) {
		coder.addDecodeType(DecodeSP);
		start = clock();
		coder.decode(postCode, newSrcCode, srcLength, DecodeSP);
		end = clock();
		decodeTime = (double) (end - start) / CLOCKS_PER_SEC;
		cout << "SP:" << decodeTime << endl;
	} else if (!strcmp(argv[4], "MS")) {
		coder.addDecodeType(DecodeMS);
		start = clock();
		coder.decode(postCode, newSrcCode, srcLength, DecodeMS);
		end = clock();
		decodeTime = (double) (end - start) / CLOCKS_PER_SEC;
		cout << "MS:" << decodeTime << endl;
	} else if (!strcmp(argv[4], "CPU")) {
		coder.addDecodeType(DecodeCPU);
		start = clock();
		start = clock();
		coder.decode(postCode, newSrcCode, srcLength, DecodeCPU);
		end = clock();
		decodeTime = (double) (end - start) / CLOCKS_PER_SEC;
		cout << "CPU:" << decodeTime << endl;
	} else if (!strcmp(argv[4], "TDMP")) {
		coder.addDecodeType(DecodeTDMP);
		start = clock();
		coder.decode(postCode, newSrcCode, srcLength, DecodeTDMP);
		end = clock();
		decodeTime = (double) (end - start) / CLOCKS_PER_SEC;
		cout << "TDMP:" << decodeTime << endl;
	} else if (!strcmp(argv[4], "TDMPCL")) {
		coder.addDecodeType(DecodeTDMPCL);
		start = clock();
		coder.decode(postCode, newSrcCode, srcLength, DecodeTDMPCL);
		end = clock();
		decodeTime = (double) (end - start) / CLOCKS_PER_SEC;
		cout << "TDMPCL:" << decodeTime << endl;
	} else if (!strcmp(argv[4], "MSCL")) {
		coder.addDecodeType(DecodeMSCL);
		start = clock();
		coder.decode(postCode, newSrcCode, srcLength, DecodeMSCL);
		end = clock();
		decodeTime = (double) (end - start) / CLOCKS_PER_SEC;
		cout << "MSCL:" << decodeTime << endl;
	}
	errNum = 0;
	for (int i = 0; i < srcLength; ++i) {
		if (srcCode[i] != newSrcCode[i])
			++errNum;
	}
	cout << "ErrNum=" << errNum << endl;
	throughPut = srcLength / decodeTime;
	cout << "ThroughPut=" << throughPut << endl;

	free(srcCode);
	free(priorCode);
	free(postCode);
	free(newSrcCode);
}
