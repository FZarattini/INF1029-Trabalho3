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
void matrix_mult(int hA, int wA, int hB, int wB, float * d_a, float * d_b, float * d_c){
    int i, j, k, index, passo;
    int line, column;
    index = blockIdx.x * blockDim.x + threadIdx.x;
    passo = gridDim.x * blockDim.x;

    for(i = index; i< hA*wB; i+= passo)
    {
    	d_c[i] = 0;
	for(j = 0; j< wA; j++)
	{
		d_c[i] += d_a[wA*(i/hA) + j] * d_b[(i%hA) + j*wB];
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

    int hA = matrixA->height;
    int wA = matrixA->width;
    int hB = matrixB->height;
    int wB = matrixB->width;

    if(matrixA == NULL || matrixB == NULL || matrixC == NULL){
    	return 0;
    }
    
    if(wA != hB)
    {
    	return 0;
    }

    qtdBlocks = (matrixC->height * matrixC->width + THREAD_QTD - 1) / THREAD_QTD;

    matrix_mult<<<qtdBlocks, THREAD_QTD>>>(hA, wA, hB, wB, matrixA->d_rows, matrixB->d_rows, matrixC->d_rows);
    
    return 1;
}
