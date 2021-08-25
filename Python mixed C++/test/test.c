#include <stdio.h>
#include <stdlib.h>
// 打印数据
void mypri(double** matrix, int r, int c)
{
    for(int i = 0; i < r; i++)
    {
        for(int j = 0; j < c; j++)
        {
            printf("%f ", matrix[i*c + j]);
        }
        printf("\n");
    }
}
// 分配空间


double *matmul(double *matrix1, double *matrix2, int r1, int c1, int r2, int c2)
{
    double *matrix; //New matrix
    matrix = (double*)malloc(sizeof(double)*(r1*c2));
    double temp = 0;
    for(int i = 0; i < r1; i++)
    {
        for(int j = 0; j < c2; j++)
        {
            temp = 0;
            for(int k = 0; k < c1; k ++)
            {
                temp = temp + matrix1[i*c1 + k]*matrix2[k*c2 + j];
            }
            matrix[i*c2 + j]= temp;
        }
    }
    return matrix;
}

int main()
{
    int r1 = 1, c1 = 1, r2 = 1, c2 = 1;
    double *matrix;
    double *matrix1;
    double *matrix2;
    // 分配内容
    matrix1= (double*)malloc(sizeof(double)*c1*r1);

    matrix2 = (double*)malloc(sizeof(double)*c2*r2);
    // 生成数据
    for(int i = 0; i < r1; i++)
    {
        for(int j = 0; j < c1; j++)
        {
            matrix1[i*c1 + j] = 1;
        }
    }

    for(int i = 0; i < r2; i++)
    {
        for(int j = 0; j < c2; j++)
        {
            matrix2[i*c2 + j] = 2;
        }
    }

    matrix = matmul(matrix1,matrix2,r1,c1,r2,c2);

    // 打印数据
    // mypri(matrix1,r1,c1);
    // mypri(matrix2,r2,c2);
    mypri(matrix,r1,c2);

    // 释放内存

    free(matrix);


    free(matrix1);

    free(matrix2);

    printf("Hello World!");
    int a;
    scanf("%d");
    return 0;
}