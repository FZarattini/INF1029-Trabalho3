#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include "matrix_lib.h"

#define THREAD_QTD 256


__global__
void scalar_mult(int n, float *d_x, float scalar_value){
    int i;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int passo = gridDim.x * blockDim.x;

    for(i = index; i < n; i += passo){
    	d_x[i] = d_x[i] * scalar_value;
    }
}

__global__
void matrix_mult(matrix* matrixA, matrix* matrixB, matrix* matrixC){
	int i, j, k, index, passo;
    index = blockIdx.x * blockDim.x + theadIdx.x;
    passo = gridDim.x * blockDim.x;

    for(i = 0; i < matrixC->height * matrixC->width; i += passo){
    	for(j = 0; j < matrixA->height * matrixA->width; j++){
    		for(k = j / matrixA->width; k < matrixB->height * matrixB->width; k += matrixB->width){
    			matrixC->d_rows[i] = matrixC->d_rows[i] + (matrixA->d_rows[j] * matrixB->d_rows[k]);
    		}
    	}
    }
}

int scalar_matrix_mult(float scalar_value, struct matrix *matrix){

    int qtdBlocks;
    int height = matrix->height;
    int width = matrix->width;

    if(matrix == NULL){
    	return 0;
    }

    qtdBlocks = (height * width + THREAD_QTD - 1) / THREAD_QTD;

    scalar_mult<<<qtdBlocks, THREAD_QTD>>>(height * width, matrix->d_rows, scalar_value);

    return 1;
    
}

int matrix_matrix_mult(struct matrix *matrixA, struct matrix *matrixB, struct matrix *matrixC){
    
    int qtdBlocks;

    if(matrixA == NULL || matrixB == NULL || matrixC == NULL){
    	return 0;
    }

    qtdBlocks = (matrixC->height * matrixC->width + THREAD_QTD - 1) / THREAD_QTD;

    matrix_mul<<<qtdBlocks, THREAD_QTD>>>(matrixA, matrixB, matrixC);
    
    return 1;
}