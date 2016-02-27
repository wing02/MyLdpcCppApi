#define DataType int
#define MAX 200000
kernel void decodeInit(global int* hCols,global float* codes,global float* q0,global float* q1,global float* priorP0,global float* priorP1,const int ldpcN,const int nonZeros){
    int batchInd=get_global_id(0);
    int nodeInd=get_global_id(1);//nonZeros
    //get the thread's and hCol;
    int hCol=hCols[nodeInd];//0-ldpcN
    float code=codes[batchInd*ldpcN+hCol];
    q0[batchInd*nonZeros+nodeInd]=1-code;
    q1[batchInd*nonZeros+nodeInd]=code;
    // init the priorP,the offset is different;
    if(nodeInd<ldpcN){
        float code=codes[batchInd*ldpcN+nodeInd];
        priorP0[batchInd*ldpcN+nodeInd]=1-code;
        priorP1[batchInd*ldpcN+nodeInd]=code;
    }
}


kernel void refreshR(global int* hRows,global float* q0,global float* q1, global float* r0,global float* r1,global int* hRowFirstPtr,global int* hRowNextPtr,const int ldpcM,const int nonZeros){
    int batchInd=get_global_id(0);
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

__kernel void refreshQ(global int* hCols,global float* q0,global float* q1,global float* r0,global float* r1,global float* priorP0,global float* priorP1,global int* hColFirstPtr,global int* hColNextPtr,const int ldpcN, const int nonZeros){
    int batchInd=get_global_id(0);
    int nodeInd=get_global_id(1);//nonZeros
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

kernel void hardDecision(global int* src,global float* r0,global float* r1,global float* priorP0,global float* priorP1,global int* hColFirstPtr,global int* hColNextPtr,const int ldpcN,const int nonZeros,global int* flags){
    int batchInd=get_global_id(0);
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
            src[batchInd*ldpcN+col]=0;
        }else if (tmp0<tmp1){
            src[batchInd*ldpcN+col]=1;
        }else{
            src[batchInd*ldpcN+col]=-1;//error
        }
    }else if (col==ldpcN){//init the flags
        flags[batchInd]=0;
    }
}

kernel void checkResult(global int* src,global int* hCols,global int* hRowFirstPtr,global int* hRowNextPtr,const int ldpcM,const int ldpcN,const int nonZeros,global int* flags){
    int batchInd=get_global_id(0);
    int nodeInd=get_global_id(1);//nonZeros
    if(nodeInd<ldpcM){//nodeInd = row;
        //init the matrixM to 0;
        int result=0;
        int row=nodeInd;
        for(int ptr=hRowFirstPtr[row];ptr!=-1;ptr=hRowNextPtr[ptr]){
            //ptr is the node location;
            result+=src[batchInd*ldpcN+hCols[ptr]];
        }
        if(result%2!=0){
            atomic_inc(&flags[batchInd]);
        }
    }
}

