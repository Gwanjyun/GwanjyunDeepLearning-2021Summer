#include <stdio.h>

double *memory_leak(double *matrix, int size)
{
    double *m;
    m = (double*)malloc(sizeof(double)*size);
    for(int i = 0; i < size; i++)
    {
        m[i] = matrix[i];
    }
    return m;
}