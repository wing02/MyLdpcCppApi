/*
 * MyTest.cpp
 *
 *  Created on: Feb 15, 2016
 *      Author: wing
 */

#include<iostream>

#include "MyLdpc.h"
using namespace std;

void test2() {
	const int z = 24;
	const int ldpcN = z * 24;
	const int ldpcK = ldpcN / 6 * 5;
	const int ldpcM = ldpcN - ldpcK;
	const enum rate_type rate = rate_5_6;

	Coder coder(ldpcK, ldpcN, rate);
	srand(time(0));

	const int srcLength = 1000;
	char * srcCode = (char*) malloc(srcLength*sizeof(char));
	char * priorCode = (char*) malloc(coder.getPriorCodeLength(srcLength));
	float * postCode = (float*) malloc(
			coder.getPostCodeLength(srcLength) * sizeof(float));
	char * newSrcCode = (char*) malloc(srcLength);

	for (int i = 0; i < srcLength; i++) {
		srcCode[i] = 'a' + i % 26;
	}

	coder.forEncoder();
	coder.forDecoder(10);
	coder.encode(srcCode, priorCode, srcLength);


	coder.test(priorCode, postCode, coder.getPriorCodeLength(srcLength), 0.17);
	coder.decode(postCode, newSrcCode, srcLength);
	cout << "srcCode="<<endl;
	for (int i = 0; i < srcLength; i++) {
		cout << srcCode[i];
	}
	cout << endl;
	cout << "newSrcCode="<<endl;
	for (int i = 0; i < srcLength; i++) {
		cout << newSrcCode[i];
	}
	cout << endl;

	for (int i = 0; i < srcLength; i++) {
		if(newSrcCode[i]!=srcCode[i]){
			cout << i<<" "<<srcCode[i]<<" "<<newSrcCode[i]<<endl;
		}
	}
	free(srcCode);
	free(priorCode);
	free(postCode);
	free(newSrcCode);
}

int main() {
	test2();
}

