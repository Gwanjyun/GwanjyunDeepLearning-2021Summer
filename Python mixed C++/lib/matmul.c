#include <stdio.h>
#include <stdlib.h>
double *matmul(double *matrix1, double *matrix2, int r1, int c1, int r2, int c2)
{
    double *matrix;
    matrix = (double*)malloc(sizeof(double)*(r1*c2));
    for(int i = 0; i < r1; i++)
    {
        for(int j = 0; j < c2; j++)
        {
            matrix[i*c2 + j] = 0;
            for(int k = 0; k < c1; k ++)
            {
                matrix[i*c2 + j] = matrix[i*c2 + j] + matrix1[i*c1 + k]*matrix2[k*c2 + j];
            }
        }
    }
    return matrix;
}