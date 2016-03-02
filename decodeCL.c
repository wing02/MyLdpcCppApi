

kernel void decodeInit(global int* hCols,global float* codes,global float* q0,global float* q1,global float* priorP0,global float* priorP1,const int ldpcN,const int nonZeros,global bool* isDones){
    int batchInd=get_global_id(0);
    int nodeInd=get_global_id(1);//nonZeros
    //get the thread's and hCol;
    int hCol=hCols[nodeInd];//0-ldpcN
    float code=codes[batchInd*ldpcN+hCol];
    float tmp=exp(8*code);//Noise=0.25
    q0[batchInd*nonZeros+nodeInd]=tmp/(1+tmp);
    q1[batchInd*nonZeros+nodeInd]=1/(1+tmp);
    // init the priorP,the offset is different;
    if(nodeInd<ldpcN){
        float code=codes[batchInd*ldpcN+nodeInd];
        float tmp=exp(8*code);//Noise=0.25
        priorP0[batchInd*ldpcN+nodeInd]=tmp/(1+tmp);
        priorP1[batchInd*ldpcN+nodeInd]=1/(1+tmp);
    }
    if(nodeInd==ldpcN){
        isDones[batchInd]=0;
    }
}


kernel void refreshR(global int* hRows,global float* q0,global float* q1, global float* r0,global float* r1,global int* hRowFirstPtr,global int* hRowNextPtr,const int ldpcM,const int nonZeros,global bool* isDones){
    int batchInd=get_global_id(0);
    if(isDones[batchInd])
        return;
    int nodeInd=get_global_id(1);//nonZeros
    int hRow=hRows[nodeInd];//0-ldpcM
    //int hCol=hCols[nodeInd];//0-ldpcN
    float dTmp=1;
    for(int ptr=hRowFirstPtr[hRow];ptr!=-1;ptr=hRowNextPtr[ptr]){
        if(nodeInd==ptr)
            continue;
        //ptr is the node location;
        dTmp*=q0[batchInd*nonZeros+ptr]-q1[batchInd*nonZeros+ptr];
    }
    r0[batchInd*nonZeros+nodeInd]=(1+dTmp)/2;
    r1[batchInd*nonZeros+nodeInd]=(1-dTmp)/2;
}

kernel void refreshQ(global int* hCols,global float* q0,global float* q1,global float* r0,global float* r1,global float* priorP0,global float* priorP1,global int* hColFirstPtr,global int* hColNextPtr,const int ldpcN, const int nonZeros,global bool* flags,global bool* isDones){
    int batchInd=get_global_id(0);
    if(isDones[batchInd])
        return;
    int nodeInd=get_global_id(1);//nonZeros
    if(nodeInd==0&&flags[batchInd]==0)
        isDones[batchInd]=1;
    int hCol=hCols[nodeInd];
    float tmp0,tmp1;
    tmp0=priorP0[batchInd*ldpcN+hCol];
    tmp1=priorP1[batchInd*ldpcN+hCol];
    for(int ptr=hColFirstPtr[hCol];ptr!=-1;ptr=hColNextPtr[ptr]){
        if(nodeInd==ptr)
            continue;
        tmp0*=r0[batchInd*nonZeros+ptr];
        tmp1*=r1[batchInd*nonZeros+ptr];
    }
    q0[batchInd*nonZeros+nodeInd]=tmp0/(tmp0+tmp1);
    q1[batchInd*nonZeros+nodeInd]=tmp1/(tmp0+tmp1);
}

kernel void hardDecision(global bool* srcBool,global float* r0,global float* r1,global float* priorP0,global float* priorP1,global int* hColFirstPtr,global int* hColNextPtr,const int ldpcN,const int nonZeros,global bool* flags,global bool* isDones){
    int batchInd=get_global_id(0);
    if(isDones[batchInd])
        return;
    int col=get_global_id(1);//nonZeros
    float tmp0,tmp1;
    //only create batchSize*ldpcN thread;
    if(col<ldpcN){
        tmp0=priorP0[batchInd*ldpcN+col];
        tmp1=priorP1[batchInd*ldpcN+col];
        for(int ptr=hColFirstPtr[col];ptr!=-1;ptr=hColNextPtr[ptr]){
            tmp0*=r0[batchInd*nonZeros+ptr];
            tmp1*=r1[batchInd*nonZeros+ptr];
        }
        if(tmp0>tmp1){
            srcBool[batchInd*ldpcN+col]=0;
        }else if (tmp0<tmp1){
            srcBool[batchInd*ldpcN+col]=1;
        }
    }else if (col==ldpcN){//init the flags
        flags[batchInd]=0;
    }
}

kernel void checkResult(global bool* srcBool,global int* hCols,global int* hRowFirstPtr,global int* hRowNextPtr,const int ldpcM,const int ldpcN,const int nonZeros,global bool* flags,global bool* isDones){
    int batchInd=get_global_id(0);
    if(isDones[batchInd])
        return;
    int nodeInd=get_global_id(1);//nonZeros
    if(nodeInd<ldpcM){//nodeInd = row;
        //init the matrixM to 0;
        bool result=0;
        for(int ptr=hRowFirstPtr[nodeInd];ptr!=-1;ptr=hRowNextPtr[ptr]){
            //ptr is the node location;
            if(srcBool[batchInd*ldpcN+hCols[ptr]])
                result ^=1;
        }
        if(flags[batchInd])
            return;
        if(result){
            flags[batchInd]=1;
        }
    }

}

//SM----------------------------------------------------------------------------------------------------------------------------

//decodeInit MS 
kernel void decodeInitMS(global int* hCols,global float* postCodes,global float* lQ,const int ldpcN,const int nonZeros,global bool* isDones){
    int batchInd=get_global_id(0);
    int nodeInd=get_global_id(1);//nonZeros
    //get the thread's and hCol;
    int hCol=hCols[nodeInd];//0-ldpcN
    float code=postCodes[batchInd*ldpcN+hCol];
    lQ[batchInd*nonZeros+nodeInd]=code;
    // init the priorP,the offset is different;
    if(nodeInd==0)
        isDones[batchInd]=0;
}

kernel void refreshRMS(global int* hRows,global float* lQ, global float* lR,global int* hRowFirstPtr,global int* hRowNextPtr,const int nonZeros,global bool* isDones){
    int batchInd=get_global_id(0);
    if(isDones[batchInd])
        return;
    int nodeInd=get_global_id(1);//nonZeros
    int hRow=hRows[nodeInd];//0-ldpcM
    //int hCol=hCols[nodeInd];//0-ldpcN
    bool a=0;
    float b=1000;
    for(int ptr=hRowFirstPtr[hRow];ptr!=-1;ptr=hRowNextPtr[ptr]){
        if(nodeInd==ptr)
            continue;
        //ptr is the node location;
        if(lQ[batchInd*nonZeros+ptr]<0)
            a=a^1;
        b=fmin(b,fabs(lQ[batchInd*nonZeros+ptr]));
    }
    if(a)
        lR[batchInd*nonZeros+nodeInd]=-b;
    else
        lR[batchInd*nonZeros+nodeInd]=b;
}

kernel void refreshPostPMS(global bool* srcBool,global float* lR,global float* postCodes,global float* lPostP,global int* hColFirstPtr,global int* hColNextPtr,const int ldpcN,const int nonZeros,global bool* flags,global bool* isDones){
    int batchInd=get_global_id(0);
    if(isDones[batchInd])
        return;
    int nodeInd=get_global_id(1);//nonZeros ,only need ldpcN
    float tmp;
    //only create batchSize*ldpcN thread;
    if(nodeInd<ldpcN){
        tmp=postCodes[batchInd*ldpcN+nodeInd];
        for(int ptr=hColFirstPtr[nodeInd];ptr!=-1;ptr=hColNextPtr[ptr]){
            tmp+=lR[batchInd*nonZeros+ptr];
        }
        if(tmp>0){
            srcBool[batchInd*ldpcN+nodeInd]=0;
        }else{
            srcBool[batchInd*ldpcN+nodeInd]=1;
        }
        lPostP[batchInd*ldpcN+nodeInd]=tmp;
    }else if (nodeInd==ldpcN){//init the flags
        flags[batchInd]=0;
    }

    
}


//before refreshQ, check the batchInd is done or not;
kernel void refreshQMS(global int* hCols,global float* lQ,global float* lR,global float* lPostP,const int ldpcN, const int nonZeros,global bool* flags,global bool* isDones){
    int batchInd=get_global_id(0);
    if(isDones[batchInd])
        return;
    int nodeInd=get_global_id(1);//nonZeros
//checkDone
    if(nodeInd==0&&flags[batchInd]==0)
        isDones[batchInd]=1;

    int hCol=hCols[nodeInd];
    lQ[batchInd*nonZeros+nodeInd]=lPostP[batchInd*ldpcN+hCol]-lR[batchInd*nonZeros+nodeInd];
}

kernel void toChar(global bool* srcBool,global char* srcCode,const int ldpcN,const int ldpcK){
    int batchInd=get_global_id(0);
    int nodeInd=get_global_id(1);//nonZeros
    if(nodeInd<ldpcK/8){
        srcCode[batchInd*ldpcK/8+nodeInd]=0;
        for(int bitOffset=0;bitOffset<8;++bitOffset){
            if(srcBool[batchInd*ldpcN+nodeInd*8+bitOffset]){
                srcCode[batchInd*ldpcK/8+nodeInd] |=(1<<bitOffset);
            }
        }
    }
}


//TDMP-------------------------------------------------------------------------------------------
kernel void decodeInitTDMP(global int* hCols,global float* postCodes,global float* lQ,const int ldpcN,const int nonZeros,global bool* isDones,global float* lPostP,const int blockHeavy,global float* lR){
    int batchInd=get_global_id(0);
    int nodeInd=get_global_id(1);//ldpcN+1
    //get the thread's and hCol;
    if(nodeInd<blockHeavy){
        int hCol=hCols[nodeInd];//0-ldpcN
        float code=postCodes[batchInd*ldpcN+hCol];
        lQ[batchInd*blockHeavy+nodeInd]=code;
    }
    // init the priorP,the offset is different;
    if(nodeInd<ldpcN){
        float code=postCodes[batchInd*ldpcN+nodeInd];
        lPostP[batchInd*ldpcN+nodeInd]=code;

        //set lr to zero
        for(int i=nodeInd;i<nonZeros;i+=ldpcN)
            lR[batchInd*nonZeros+i]=0;
    }
    else if(nodeInd==ldpcN){
        isDones[batchInd]=0;
    }

}

kernel void refreshRTDMP(global int* hRows,global float* lQ, global float* lR,global int* hRowFirstPtr,global int* hRowNextPtr,const int nonZeros,global bool* isDones,const int blockHeavy,const int offset){
    int batchInd=get_global_id(0);
    if(isDones[batchInd])
        return;
    int nodeInd=get_global_id(1);//blockHeavy
    if(nodeInd<blockHeavy){
        int newNodeInd=nodeInd+offset;
        int hRow=hRows[newNodeInd];

        bool a=0;
        float b=1000;
        for(int ptr=hRowFirstPtr[hRow];ptr!=-1;ptr=hRowNextPtr[ptr]){
            if(newNodeInd==ptr)
                continue;
            //ptr is the node location;
            if(lQ[batchInd*blockHeavy+ptr-offset]<0)
                a=a^1;
            b=fmin(b,fabs(lQ[batchInd*blockHeavy+ptr-offset]));
        }
        if(a)
            lR[batchInd*nonZeros+newNodeInd]=-b;
        else
            lR[batchInd*nonZeros+newNodeInd]=b;
    }
    //if(nodeInd==0&& batchInd==0){
    //    printf("offset=%d\n",offset);
    //    for(int i=0;i<100;++i)
    //        printf("%f ",lR[i]);
    //    printf("Refresh R\n");
    //}
}

kernel void refreshPostPTDMP(global float* lR,global float* lQ,global float* lPostP,global int* hCols,const int ldpcN,const int nonZeros,global bool* isDones,const int blockHeavy,const int offset){
    int batchInd=get_global_id(0);
    if(isDones[batchInd])
        return;
    int nodeInd=get_global_id(1);//threadNum , <blockHeavy
    int hCol=hCols[nodeInd+offset];
    //only create batchSize*ldpcN thread;
    lPostP[batchInd*ldpcN+hCol]=lQ[batchInd*blockHeavy+nodeInd]+lR[batchInd*nonZeros+nodeInd+offset];
        
}

kernel void hardDecisionTDMP(global float* lPostP,global bool* srcBool,global bool* flags,const int ldpcN,global bool* isDones){
    int batchInd=get_global_id(0);
    if(isDones[batchInd])
        return;
    int nodeInd=get_global_id(1);//nonZeros ,only need ldpcN
    float tmp;
    //only create batchSize*ldpcN thread;
    if(nodeInd<ldpcN){
        tmp=lPostP[batchInd*ldpcN+nodeInd];
        if(tmp>0){
            srcBool[batchInd*ldpcN+nodeInd]=0;
        }else if(tmp<0){
            srcBool[batchInd*ldpcN+nodeInd]=1;
        }else if(tmp==0){
            //printf("Error in hardDecision\n");
        }
    }else if (nodeInd==ldpcN){//init the flags
        flags[batchInd]=0;
    }
}


kernel void refreshQTDMP(global int* hCols,global float* lQ,global float* lR,global float* lPostP,const int ldpcN, const int nonZeros,global bool* flags,global bool* isDones,const int blockHeavy,const int offset){
    int batchInd=get_global_id(0);
    if(isDones[batchInd])
        return;
    int nodeInd=get_global_id(1);//nonZeros
    int newNodeInd=nodeInd+offset;
    int hCol=hCols[newNodeInd];
    lQ[batchInd*blockHeavy+nodeInd]=lPostP[batchInd*ldpcN+hCol]-lR[batchInd*nonZeros+newNodeInd];
}



kernel void checkDones(global bool*flags,global bool*isDones){
    int batchInd=get_global_id(0);
    if(flags[batchInd]==0)
        isDones[batchInd]=1;
}
