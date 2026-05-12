#include <iostream>
#include <cmath>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define TILE_SIZEK 8
#define TILE_SIZE 16
#define PADDING 1

__global__ void matrixMul(float* a, float* b, float* c, int aM, int aN, int bM, int bN, int tile_size) {

    __shared__ alignas(32) float tile_a[TILE_SIZEK][TILE_SIZE][TILE_SIZE + PADDING];
    __shared__ alignas(32) float tile_b[TILE_SIZEK][TILE_SIZE][TILE_SIZE + PADDING];

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    float fp16zero = 0.0f;
    float Acumm = 0.0f;



    // loop externo percorre a dimensao N em blocos
    // a cada iteração carrega sub-tiles para a shared memory

    for (int i = 0; i < ((aN + tile_size - 1) / tile_size); i++) {



        for (int j = 0; j < TILE_SIZEK; j++) {

            int delta = j * TILE_SIZE;

            int tile_row = tile_size * i + delta + threadIdx.y;
            int tile_col = tile_size * i + delta + threadIdx.x;


            if (row < aM && tile_col < aN) {
                tile_a[j][threadIdx.y][threadIdx.x] = a[row * aN + tile_col];
            }
            else {
                tile_a[j][threadIdx.y][threadIdx.x] = fp16zero;
            }

            if (tile_row < bM && col < bN) {
                tile_b[j][threadIdx.y][threadIdx.x] = b[tile_row * bN + col];
            }
            else {
                tile_b[j][threadIdx.y][threadIdx.x] = fp16zero;
            }
        }

        __syncthreads();

        // ideia inicial era acumular parcialmente o resultado num fp16
        // e depois acrescentar, aos poucos, ao acumulador global fp32
        // era mais rapido, porem a perda de precisao fez com que eu revertesse a ideia
        // porem mantive o loop dividido

        for (int j = 0; j < TILE_SIZEK; j++) {
            float halfAcumm = fp16zero;
            for (int k = 0; k < TILE_SIZE; k++) {
                // fmaf = fused multiply add
                halfAcumm = fmaf(tile_a[j][threadIdx.y][k], tile_b[j][k][threadIdx.x], halfAcumm);
            }
            Acumm += halfAcumm;

        }

        __syncthreads();

    }

    if (row < aM && col < bN) {
        c[row * bN + col] = Acumm;
    }

}

int main() {

    int m;

    scanf("%d", &m);

    int n = m;

    size_t bytes = n * m * sizeof(float);

    float* h_a, * h_b, * h_c;
    float* d_a, * d_b, * d_c;

    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);

    size_t pitchA, pitchB, pitchC;

    //endereço da memoria ado inicio, pitch, tamanho da stride em bytes, quantidade de linhas
    cudaMallocPitch(&d_a, &pitchA, n * sizeof(float), n);
    cudaMallocPitch(&d_b, &pitchB, n * sizeof(float), n);
    cudaMallocPitch(&d_c, &pitchC, n * sizeof(float), n);

    //inicializar a matriz
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) scanf("%f", &h_a[i * n + j]);
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) scanf("%f", &h_b[i * n + j]);
    }

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 block_cnt((n + block_size.x - 1) / block_size.x, (n + block_size.x - 1) / block_size.x);

    matrixMul << <block_cnt, block_size >> > (d_a, d_b, d_c, n, n, n, n, TILE_SIZE * TILE_SIZEK);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f", h_c[i * n + j]);
            if (j == (n - 1)) printf("\n");
            else printf(" ");
        }
    }


    free(h_a);
    free(h_b);
    free(h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}