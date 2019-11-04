#include <stdio.h>
#include "matrix_lib.h"

struct matrix matrixA, matrixB, matrixC;

int load_matrix(struct matrix *matrix, char *filename) {
        unsigned long int i = 0;
        unsigned long int N = 0;
        FILE *fd = NULL;

        /* Check the numbers of the elements of the matrix */
        N = matrix->height * matrix->width;

        /* Check the integrity of the matrix */
        if (N == 0 || matrix->rows == NULL) return 0;

        /* Try to open file of floats */
        if ((fd = fopen(filename, "rb")) == NULL) {
            printf("Unable to open file %s\n", filename);
            return 0;
        }

        float *nxt_a = matrix->h_rows; 

        for ( i = 0;
	        i < N; 
	        i += 8, nxt_a += 8) {

	        if (fread(nxt_a, sizeof(float), 8, fd) != 8) {
                printf("Error reading from file %s: short read (less than 8 floats)\n", filename);
                return 0;
	        }
        }

        if (fd != NULL) fclose(fd);

        return 1;
}

int main(void){
    unsigned long int DimA_M, DimA_N, DimB_M, DimB_N;
    char *matrixA_filename, *matrixB_filename, *result1_filename, *result2_filename;
    char *eptr = NULL;
    cudaError_t cudaError;

    // Check arguments
    if (argc != 10) {
            printf("Usage: %s <scalar_value> <DimA_M> <DimA_N> <DimB_M> <DimB_N> <matrixA_filename> <matrixB_filename> <result1_filename> <result2_filename>\n", argv[0]);
            return 0;
    } else {
            //printf("Number of args: %d\n", argc);
            //for (int i=0; i<argc; ++i)
            //       printf("argv[%d] = %s\n", i, argv[i]);
    }

    // Convert arguments
    scalar_value = strtof(argv[1], NULL);
    DimA_M = strtol(argv[2], &eptr, 10);
    DimA_N = strtol(argv[3], &eptr, 10);
    DimB_M = strtol(argv[4], &eptr, 10);
    DimB_N = strtol(argv[5], &eptr, 10);
    matrixA_filename = argv[6];
    matrixB_filename = argv[7];
    result1_filename = argv[8];
    result2_filename = argv[9];

    if ((scalar_value == 0.0f) || (DimA_M == 0) || (DimA_N == 0) || (DimB_M == 0) || (DimB_N == 0)) {
            printf("%s: erro na conversao do argumento: errno = %d\n", argv[0], errno);

            /* If a conversion error occurred, display a message and exit */
            if (errno == EINVAL)
            {
                printf("Conversion error occurred: %d\n", errno);
                return 1;
            }

            /* If the value provided was out of range, display a warning message */
            if (errno == ERANGE) {
                printf("The value provided was out of rangei: %d\n", errno);
                return 1;
        }
    }

    /* Allocate the arrays of the four matrixes */
    float *a=  (float*)aligned_alloc(32, DimA_M*DimA_N*sizeof(float));
    float *b = (float*)aligned_alloc(32, DimB_M*DimB_N*sizeof(float));
    float *c = (float*)aligned_alloc(32, DimA_M*DimB_N*sizeof(float));

    if ((a == NULL) || (b == NULL) || (c == NULL)) {
        printf("%s: array allocation problem.", argv[0]);
        return 1;
    }

    float *h_a=  (float*)malloc(DimA_M * DimA_N * sizeof(float));
    float *h_b = (float*)malloc(DimB_M * DimB_N * sizeof(float));
    float *h_c = (float*)malloc(DimA_M * DimB_N * sizeof(float));


    float *d_a;
    float *d_b;
    float *d_c;

    cudaError = cudaMalloc(&d_a, DimA_M * DimA_N * sizeof(float));
    cudaError = cudaMalloc(&d_b, DimB_M * DimB_N * sizeof(float));
    cudaError = cudaMalloc(&d_c, DimA_M * DimB_N * sizeof(float));

    matrixA.height = DimA_M;
    matrixA.width = DimA_N;
    matrixA.h_rows = h_a;
    matrixA.d_rows = d_a;

    if (!load_matrix(&matrixA, matrixA_filename)) {
        printf("%s: matrixA initialization problem.", argv[0]);
        return 1;
    }
        
    matrixB.height = DimB_M;
    matrixB.width = DimB_N;
    matrixB.h_rows = h_b;
    matrixB.d_rows = d_b;

    if (!load_matrix(&matrixB, matrixB_filename)) {
        printf("%s: matrixB initialization problem.", argv[0]);
        return 1;
    }

    matrixC.height = DimA_M;
    matrixC.width = DimB_N;
    matrixC.h_rows = h_c;
    matrixC.d_rows = d_c;

    cudaError = cudaMemcpy(d_a, h_a, DimA_M * DimA_N * sizeof(float), cudaMemcpyHostToDevice);

    cudaError = cudaMemcpy(d_b, h_b, DimB_M * DimB_N * sizeof(float), cudaMemcpyHostToDevice);

    cudaError = cudaMemcpy(d_c, h_c, DimA_M * DimB_N * sizeof(float), cudaMemcpyHostToDevice);
    

    return 1;
}