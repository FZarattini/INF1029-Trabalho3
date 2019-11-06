#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include "matrix_lib.h"

#define THREAD_QTD 256;


__global__
void mult(int n, float *d_x, float scalar_value){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int passo = gridDIm.x * blockDim.x;

    for(i = index; i < n; i += passo){
    	d_x[i] = d_x[i] * scalar_value;
    }
}

int scalar_matrix_mult(float scalar_value, struct matrix *matrix){
    if(matrix == null){
    	return 0;
    }

    int qtdBlocks = (matrix->height * matrix->width + THREAD_QTD - 1) / THREAD_QTD;

    mul<<<qtdBlocks, THREAD_QTD>>>(matrix->height * matrix->width, matrix->d_rows, scalar_value);

    return 1;
    
}

int matrix_matrix_mult(struct matrix *matrixA, struct matrix *matrixB, struct matrix *matrixC){

}