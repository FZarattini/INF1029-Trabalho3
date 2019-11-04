#include <stdio.h>
#include <cuda_runtime.h>
#include "matrix_lib.h"]


__global__
void mult(int n, float *d_x, float *d_y){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < n){
        d_y[index] = d_x[index] * d_y[index];
    }
}

int scalar_matrix_mult(float scalar_value, struct matrix *matrix){
    
}

int matrix_matrix_mult(struct matrix *matrixA, struct matrix *matrixB, struct matrix *matrixC){

}