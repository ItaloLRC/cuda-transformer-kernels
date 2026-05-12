#include <iostream>
#include <cmath>
#include <cuda_fp16.h>
#include <mma.h>
#include <algorithm>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define MAXREG 16

#define ROWSH 1
#define MAXN 4096

#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        std::cerr << "ERRO CUDA: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}



// P = proximo pow2 >= ceil(N * K / 100)
// O kernel recebe P ja calculado pelo main, nao K diretamente
// Bitonic sort exige tamanho potencia de 2 — P2 e esse valor
__global__ void meu_kernel(float* a, int M, int N, int K, int P) {

    //__shared__ float shmem[MAXN];

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int qtdV = (N + blockDim.x - 1) / blockDim.x;
    int Nmem = N;

    int tx = threadIdx.x;



    __syncthreads();


    // SORT PARCIAL Bitonic Sort parcial sobre os primeiros P elementos
    // Ordena N elementos inteiros em segmentos bitônicos de tamanho P.
    // Ao final, cada bloco de P elementos esta ordenado alternadamente
    // (crescente/decrescente), pronto para a fase de reducao.
    // Loop externo: dobra o tamanho do segmento bitônico a cada iteracao (len = 1, 2, 4 ... P/2).

    for (int len = 1; len < P; len <<= 1) {
        int dir = len << 1; // controla a direçao


        // Loop interno : realiza as comparacoes e trocas da etapa atual do Bitonic sort
        // inc vai de len ate 1, reduzindo o gap de comparacao a cada passo.
        for (int inc = len; inc > 0; inc >>= 1) {

            for (int l = 0; l < N; l += blockDim.x) {

                int tid = (tx + l);

                // Calcula o par (i, i+inc) que esta thread vai comparar.
                // low = posicao dentro do bloco de inc elementos.
                // i = indice do elemento da esquerda do par.

                int low = tid & (inc - 1);
                int i = (tid << 1) - low;

                if (i < N && (i + inc) < N) {
                    // reverse: define se o segmento atual deve ficar crescente ou decrescente.
                    // Baseado no bit de 'dir' na posicao 'i', alterna a direcao por segmento.
                    bool reverse = ((dir & i) == 0);

                    float x0 = a[row * Nmem + i];
                    float x1 = a[row * Nmem + i + inc];

                    // swap quando a ordem esta errada para a direcao desejada.
                    bool swap = reverse ^ (x0 > x1);

                    if (swap) {
                        a[row * Nmem + i] = x1;
                        a[row * Nmem + i + inc] = x0;
                    }
                }
            }

            __syncthreads();
        }
    }

    int it = 0;

    // fase 2, reduçao iterativa para isolar os P maiores
    // A cada iteracao compara pares de segmentos de tamanho P
    // mantém apenas os maiores descartando metade dos candidatos por passo
    // Evita ordenar N inteiro, so ordena o necessario para extrair o top-K.

    for (int stride = P; stride < N; stride <<= 1) {

        int newT = (N >> it);// tamanho do subarray ativo nesta iteracao

        // Compara elemento idx com elemento idx+P dentro de cada par de segmentos.
        // fmaxf descarta o menor dos dois, mantendo so os candidatos ao top-K.

        for (int i = 0; i < (newT >> 1); i += blockDim.x) {

            int t = (tx + i);

            int low = t & (P - 1);
            int idx = (t << 1) - low;


            if (idx < newT && (idx + P) < newT) {
                a[row * Nmem + idx] = fmaxf(a[row * Nmem + idx], a[row * Nmem + idx + P]);
            }

        }

        __syncthreads();

        // Compactacao: move os vencedores para posicoes contíguas.
        // Sem isso o proximo passo operaria sobre buracos no array.
        // src = i + (i/P)*P pula os slots descartados do passo anterior.

        for (int i = tx; i < (N >> (it + 1)); i += blockDim.x) {

            int dst = i;
            int src = i + ((i / P) * P);

            if (src < N && dst < N) {
                a[row * Nmem + dst] = a[row * Nmem + src];
            }
        }

        __syncthreads();



        // Reordena o subarray compactado com Bitonic Sort parcial (tamanho P/2 -> P)..
        int len = P >> 1;
        int dir = len << 1;

        for (int inc = len; inc > 0; inc >>= 1) {
            for (int l = 0; l < (N >> (it + 1)); l += blockDim.x) {

                int tid = (tx + l);

                int low = tid & (inc - 1);
                int i = (tid << 1) - low;

                if (i < N && (i + inc) < N) {
                    bool reverse = ((dir & i) == 0);

                    float x0 = a[row * Nmem + i];
                    float x1 = a[row * Nmem + i + inc];


                    bool swap = reverse ^ (x0 > x1);

                    if (swap) {
                        a[row * Nmem + i] = x1;
                        a[row * Nmem + i + inc] = x0;
                    }
                }
            }
        }

        it++;
    }

}
// cpu para validaçao
void initMatrix(float* a, int M, int N) {
    for (int i = 0; i < M; i++) {
        int it = 0;
        for (int j = 0; j < N; j++) {
            it++;
            a[i * N + j] = (float)it / 1000.0f;
        }
    }
}

void check(float* a, float* b, int M, int N, int P2, int P) {

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            if (a[i * N + j] != b[i * N + j]) {
                printf("ERRO\n");
                printf("%f != %f\n", a[i * N + j], b[i * N + j]);
                printf("%d %d\n", i, j);
                return;
            }
        }
    }

    printf("OK\n");

}

int main() {
    int K, M, N, P, P2;

    scanf("%d%d%d", &K, &M, &N);
    // arredondando para cima sem usar o ceil
    P = (N * K + 99) / 100;


    // P2 = proxima potencia de 2 maior que P
    for (int i = 1;; i <<= 1) {
        if (i >= P) {
            P2 = i;
            break;
        }
    }

    size_t bytes = M * N * sizeof(float);
    size_t buffer_size = 16 * 1024 * 1024;


    char* buffer;
    float* h_in, * h_out;
    float* d_in;

    buffer = (char*)malloc(buffer_size);
    h_in = (float*)malloc(bytes);
    h_out = (float*)malloc(bytes);

    cudaMalloc(&d_in, bytes);

    char* ptr = buffer;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) scanf("%f", &h_in[i * N + j]);
    }

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    dim3 blockSize(256, 1);
    dim3 blockCnt(1, (M + blockSize.y - 1) / blockSize.y);

    meu_kernel << <blockCnt, blockSize >> > (d_in, M, N, K, P2);

    cudaMemcpy(h_out, d_in, bytes, cudaMemcpyDeviceToHost);
    // Imprime so os P primeiros elementos de cada linha
    // Buffer de 16MB com fwrite evita overhead de printf para matrizes grandes.
    for (int i = 0; i < M; i++) {

        for (int j = 0; j < P; j++) {

            if (ptr + 25 > buffer + buffer_size) {
                fwrite(buffer, 1, ptr - buffer, stdout);
                ptr = buffer;
            }

            ptr += snprintf(ptr, 20, "%.3f", h_out[i * N + j]);
            if (j == (P - 1)) *ptr++ = '\n';
            else *ptr++ = ' ';

        }
    }

    if (ptr > buffer) {
        fwrite(buffer, 1, ptr - buffer, stdout);
    }


    return 0;
}