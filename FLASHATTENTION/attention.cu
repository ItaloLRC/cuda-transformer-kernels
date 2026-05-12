#include <iostream>
#include <cmath>
#include <mma.h>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// como o flsatt tem 3 partes gemm1 softmax e gemm2, tenho que buscar dividir esses 2 gemms em tiles, para que reduza o uso da shmem, mesmo que use mais iteraçoes!!

#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        std::cerr << "ERRO CUDA: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// VALX representa linhas processadas por bloco
// Valor pequeno (8*256) para caber na shmem de minha placa de video (GTX1650)

#define VALX 8

// Colunas carregadas na shmem
#define VALY 256
// padding para evitar bankconflict
#define MAXL 4097
#define PADDING 1

#define MASK 0xffffffff

__global__ void meuAttention(float* Q, float* K, float* V, int N, int M,
    float soft_scale, float* l, float* m, float* O) {


    int Bc = blockDim.x; // coluna
    int Br = blockDim.y; // linha

    // quantas iteraçoes necessarias para cobrir VALY colunas
    // usando aritmetica de inteiros para realizar um ceil
    int bound_Y = (VALY + Bc - 1) / Bc;

    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= M) return;
    // Stride alinhado em 4 para tornar viavel a leitura em float4
    int N4 = ((N + 3) / 4) * 4;
    //int N4 = N;

    // inicialização das matrizes parciais na shmem
    // MATRIZES Q, K E V
    __shared__ float Qi[VALX][VALY];
    __shared__ float Kj[VALX][VALY];
    __shared__ float Vj[VALX][VALY];
    __shared__ float S[VALX][VALX];

    // buffer intermediario para reduçao
    __shared__ float red[VALX][VALX];

    // row_mx e row_s: maximo e soma por linha do tile.
    // Necessarios para o softmax online
    __shared__ float row_mx[VALX];
    __shared__ float row_s[VALX];

    // loop externo percorre a dimensao M de K e V

    for (int i = 0; i < (M + Br - 1) / Br; i++) {

        // inicialização
        if (threadIdx.x == 0) {
            row_mx[threadIdx.y] = -1e9f; // -inf para o max online do softmax
            row_s[threadIdx.y] = 0.0f;
        }


        if (threadIdx.x < VALX) {
            S[threadIdx.y][threadIdx.x] = 0.0f;
            red[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int k_start = 0;
        //int k_start = 0; k_start < (N + VALY - 1) / VALY; k_start++
        // 
        // SIMD = Single Instruction Multiple Data
        // SIMD, percebi que float4 era melhor que float2
        // que por sua vez é melhor que o float
        // pelo menos na GTX1650
        // ponteiros para interpretar em float4
        float4* K4 = reinterpret_cast<float4*>(K);
        float4* V4 = reinterpret_cast<float4*>(V);
        float4* Q4 = reinterpret_cast<float4*>(Q);

        //FASE 1: calculo de S = Q * K^T / sqrt(dk)

        // Caminho rapido: tiles completos, N divisivel por VALY.
        // sem verificaçao de bounds
        // pois o loop nao itera ate o fim (floor de N/VALY)

        for (; k_start < N / VALY; k_start++) {
            int delta = k_start * VALY;

            int t_col = threadIdx.x;

            // pegando os pacotes de 4 em 4
            // carrega 4 floats de K e Q por thread em uma unica instruçao de memoria
            float4 Kji = K4[((i * Br) + threadIdx.y) * ((N + 3) / 4) + t_col + delta / 4];
            float4 Qji = Q4[row * ((N + 3) / 4) + t_col + delta / 4];

            // colocando eles na shmem
            Kj[threadIdx.y][t_col * 4] = Kji.x;
            Kj[threadIdx.y][t_col * 4 + 1] = Kji.y;
            Kj[threadIdx.y][t_col * 4 + 2] = Kji.z;
            Kj[threadIdx.y][t_col * 4 + 3] = Kji.w;

            Qi[threadIdx.y][t_col * 4] = Qji.x;
            Qi[threadIdx.y][t_col * 4 + 1] = Qji.y;
            Qi[threadIdx.y][t_col * 4 + 2] = Qji.z;
            Qi[threadIdx.y][t_col * 4 + 3] = Qji.w;

            // sincronização para evitar raceconditions e eventuais erros
            __syncthreads();


            // feito a operaçoa de produto escalar de Q e K
            for (int y = 0; y < Br; y++) {
                float sum = 0.0f;

                for (int x = 0; x < bound_Y; x++) {
                    int t_col = (Bc * x + threadIdx.x);
                    // a operaçao fmaf e mais rapida
                    // fused multiply add
                    sum = fmaf(Qi[threadIdx.y][t_col], Kj[y][t_col], sum);
                }

                // constante
                // divide antes da reduçao para evitar overflow na soma acumulada
                sum /= soft_scale;

                // reduçao com warp shuffle
                // com o __shfl_down_sync
                // somar todos os 32 valores do warp
                for (int x = 16; x > 0; x /= 2) {
                    sum += __shfl_down_sync(MASK, sum, x, 32);
                }

                // thread lider de cada warp deposita a soma parcial em red
                if (threadIdx.x % 32 == 0) {
                    red[threadIdx.y][threadIdx.x / 32] = sum;
                }

                __syncthreads();

                // thread 0 consolida as somas dos warps e atualiza S e row_mx.
                // row_mx rastreia o maximo online para estabilidade do softmax
                // sem precisar de uma segunda passagem sobre S
                if (threadIdx.x == 0) {
                    float csum = 0.0f;

                    for (int x = 0; x < Bc / 32; x++) {
                        csum += red[threadIdx.y][x];
                        red[threadIdx.y][x] = 0.0f;
                    }

                    S[threadIdx.y][y] = csum + S[threadIdx.y][y];
                    float aux = S[threadIdx.y][y];

                    if (aux > row_mx[threadIdx.y]) row_mx[threadIdx.y] = aux;
                }

            }

        }

        // faz o mesmo que o loop passado
        // porem esse daqui testa se esta fora dos limites
        // separei em 2 loops diferentes para evitar o overhead dos condicionais
        // além de viabilizar o float4

        for (; k_start < (N + VALY - 1) / VALY; k_start++) {
            int delta = k_start * VALY;

            // nao uso SIMD para esses casos
            // resulta em comportamento nao esperado

            for (int j = 0; j < bound_Y; j++) {
                int t_col = (Bc * j + threadIdx.x);

                if (((i * Br) + threadIdx.y) < M && (t_col + delta) < N) {
                    Kj[threadIdx.y][t_col] = K[((i * Br) + threadIdx.y) * N4 + t_col + delta];
                }
                else {
                    Kj[threadIdx.y][t_col] = 0.0f;
                }


                if (row < M && (t_col + delta) < N) {
                    Qi[threadIdx.y][t_col] = Q[row * N4 + t_col + delta];
                }
                else {
                    Qi[threadIdx.y][t_col] = 0.0f;
                }

            }

            __syncthreads();

            for (int y = 0; y < Br; y++) {
                float sum = 0.0f;

                for (int x = 0; x < bound_Y; x++) {
                    int t_col = (Bc * x + threadIdx.x);
                    sum = fmaf(Qi[threadIdx.y][t_col], Kj[y][t_col], sum);
                }

                sum /= soft_scale;

                for (int x = 16; x > 0; x /= 2) {
                    sum += __shfl_down_sync(MASK, sum, x, 32);
                }


                if (threadIdx.x % 32 == 0) {
                    red[threadIdx.y][threadIdx.x / 32] = sum;
                }

                __syncthreads();

                if (threadIdx.x == 0) {
                    float csum = 0.0f;

                    for (int x = 0; x < Bc / 32; x++) {
                        csum += red[threadIdx.y][x];
                        red[threadIdx.y][x] = 0.0f;
                    }

                    S[threadIdx.y][y] = csum + S[threadIdx.y][y];
                    float aux = S[threadIdx.y][y];

                    if (aux > row_mx[threadIdx.y]) row_mx[threadIdx.y] = aux;
                }

            }
        }


        // guarda o estado anterior
        float row_m_prev = m[row];
        float row_l_prev = l[row];


        __syncthreads();

        // faz-se o softmax online
        // para evitar overflow row_s acumula a soma normalizada do tile

        if (threadIdx.x == 0) {
            for (int y = 0; y < Br; y++) {
                if (row < M && (Br * i + y) < M)  S[threadIdx.y][y] = __expf(S[threadIdx.y][y] - row_mx[threadIdx.y]);
                row_s[threadIdx.y] += S[threadIdx.y][y];
            }

        }
        __syncthreads();
        // atualiza o maximo global
        float row_m_new = fmaxf(row_m_prev, row_mx[threadIdx.y]);

        // atualiza a soma normalizada
        float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_mx[threadIdx.y] - row_m_new) * row_s[threadIdx.y]);

        __syncthreads();

        int k_start2 = 0;

        // fase final do attention
        // O = S * V
        // float4 sem verificaçao de limites
        // afinal o loop impossibilita chegar no limite (floor de N/VALY)
        for (; k_start2 < N / VALY; k_start2++) {
            int delta = k_start2 * VALY;
            int t_col = threadIdx.x;
            // pega os valores com SIMD
            float4 Vij = V4[((i * Br) + threadIdx.y) * ((N + 3) / 4) + t_col + delta / 4];

            Vj[threadIdx.y][t_col * 4] = Vij.x;
            Vj[threadIdx.y][t_col * 4 + 1] = Vij.y;
            Vj[threadIdx.y][t_col * 4 + 2] = Vij.z;
            Vj[threadIdx.y][t_col * 4 + 3] = Vij.w;


            __syncthreads();
            for (int x = 0; x < bound_Y; x++) {
                int t_col = (x * Bc + threadIdx.x);
                float pv = 0.0f;
                for (int y = 0; y < Br; y++) {
                    // fused multiply add
                    pv = fmaf(S[threadIdx.y][y], Vj[y][t_col], pv);
                }

                if (row < M && (t_col + delta) < N) {
                    // correçao incremental de O reescala o valor anterior pelo
                    // fator exp(m_prev - m_new) e adiciona a contribuicao do tile atual
                    // dividindo por row_l_new para manter a normalizaçao
                    // Equivale a recalcular softmax(Q*K^T)*V sem rever tiles anteriores
                    O[row * N4 + (t_col + delta)] = (1 / row_l_new) * (row_l_prev * __expf(row_m_prev - row_m_new) * O[row * N4 + (t_col + delta)] + (__expf(row_mx[threadIdx.y] - row_m_new) * pv));
                }

            }

            __syncthreads();
        }

        // faz o mesmo que o anterior
        // porem pode resultar em out of bounds
        // por isso tem condicionais a mais testando os limites
        // não uso float4

        for (; k_start2 < (N + VALY - 1) / VALY; k_start2++) {
            int delta = k_start2 * VALY;
            // nao se usa SIMD para esses casos
            // resulta em comportamento nao esperado

            for (int j = 0; j < bound_Y; j++) {
                int t_col = (Bc * j + threadIdx.x);

                if (((i * Br) + threadIdx.y) < M && (t_col + delta) < N) {
                    Vj[threadIdx.y][t_col] = V[((i * Br) + threadIdx.y) * N4 + t_col + delta];
                }
                else {
                    Vj[threadIdx.y][t_col] = 0.0f;
                }

            }

            __syncthreads();

            for (int x = 0; x < bound_Y; x++) {
                int t_col = (x * Bc + threadIdx.x);
                float pv = 0.0f;
                for (int y = 0; y < Br; y++) {
                    pv = fmaf(S[threadIdx.y][y], Vj[y][t_col], pv);
                }

                if (row < M && (t_col + delta) < N) {

                    O[row * N4 + (t_col + delta)] = (1 / row_l_new) * (row_l_prev * __expf(row_m_prev - row_m_new) * O[row * N4 + (t_col + delta)] + (__expf(row_mx[threadIdx.y] - row_m_new) * pv));
                }

            }

            __syncthreads();
        }

        if (row < M) {
            // atribuiçao final
            l[row] = row_l_new;
            m[row] = row_m_new;
        }


    }

}

// Inicializa l=0 e m=-inf antes do loop principal.
__global__ void initLM(float* l, float* m, int M) {

    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < M) {
        l[tid] = 0.0f;
        m[tid] = -1e9f;
    }

}

// adicionar syncthreads no final de cada laço de repetiçao
// validaçao pela CPU
void initMatrix(float* a, int M, int N) {

    int N4 = ((N + 3) / 4) * 4;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N4 + j] = (rand() % 101) / 100.0;
        }
    }
}

void printMatrix(float* a, int M, int N) {
    printf("\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.4f ", a[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void MMA(float* a, float* b, float* out, int M, int N) {

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acum = 0.0f;
            for (int k = 0; k < M; k++) {
                acum += a[i * M + k] * b[k * N + j];
            }
            out[i * N + j] = acum;
        }
    }
}

void MMAt(float* a, float* b, float* out, int M, int N) {



    std::vector<float> mnew(M * N);

    // transpor matriz B
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            mnew[j * M + i] = b[i * N + j];
        }
    }

    // MMA
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            float acum = 0.0f;
            for (int k = 0; k < N; k++) {
                acum += a[i * N + k] * mnew[k * M + j];
            }
            out[i * M + j] = acum;
        }
    }
}

void softmax(float* a, float softscale, int M, int N) {

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] *= softscale;
        }
    }

    for (int i = 0; i < M; i++) {
        float mx = -1e9;
        float sumL = 0.0f;
        for (int j = 0; j < N; j++) mx = fmaxf(mx, a[i * N + j]);

        for (int j = 0; j < N; j++) {
            a[i * N + j] = expf(a[i * N + j] - mx);
            sumL += a[i * N + j];
        }

        for (int j = 0; j < N; j++) a[i * N + j] /= sumL;

    }


}
// tolerancia de 0.03
void checkMatrix(float* a, float* b, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (fabsf(a[i * N + j] - b[i * N + j]) > 0.03f) {
                printf("ERRO %d %d \n %f %f \n", i, j, a[i * N + j], b[i * N + j]);
                return;
            }
        }
    }
    printf("OK\n");
}


int main() {
    int M, N;

    scanf("%d%d", &M, &N);
    // N4: stride da matriz alinhado a multiplo de 4 para leitura com float4.
    int N4 = ((N + 3) / 4) * 4;
    //int N4 = N;

    size_t bytes = N4 * M * sizeof(float);
    size_t bytes_lm = (M + 1) * sizeof(float);
    size_t buffer_size = 16 * 1024 * 1024;

    char* buffer;
    float* h_Q, * h_K, * h_V, * h_O;
    float* d_Q, * d_K, * d_V, * d_O;
    float* d_l, * d_m;

    h_Q = (float*)malloc(bytes);
    h_K = (float*)malloc(bytes);
    h_V = (float*)malloc(bytes);
    h_O = (float*)malloc(bytes);
    buffer = (char*)malloc(buffer_size);

    cudaMalloc(&d_Q, bytes);
    cudaMalloc(&d_K, bytes);
    cudaMalloc(&d_V, bytes);
    cudaMalloc(&d_O, bytes);

    cudaMalloc(&d_l, bytes_lm);
    cudaMalloc(&d_m, bytes_lm);

    cudaMemset(d_O, 0, bytes);

    char* ptr = buffer;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) scanf("%f", &h_Q[i * N4 + j]);
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) scanf("%f", &h_K[i * N4 + j]);
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) scanf("%f", &h_V[i * N4 + j]);
    }

    cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice);

    initLM << <(M + 256 - 1) / 256, 256 >> > (d_l, d_m, M);

    dim3 block_size(64, VALX);
    dim3 block_cnt(1, (M + block_size.y - 1) / block_size.y);

    cudaDeviceSynchronize();

    float soft_scale = sqrtf(N);

    meuAttention << <block_cnt, block_size >> > (d_Q, d_K, d_V, N, M, soft_scale, d_l, d_m, d_O);

    cudaMemcpy(h_O, d_O, bytes, cudaMemcpyDeviceToHost);


    // buffer de 16MB com fwrite para evitar overhead de printf por elemento
    // para matrizes grandes (ate 4096x4096 = 16M floats).

    for (int i = 0; i < M; i++) {

        for (int j = 0; j < N; j++) {

            if (ptr + 25 > buffer + buffer_size) {
                fwrite(buffer, 1, ptr - buffer, stdout);
                ptr = buffer;
            }

            ptr += snprintf(ptr, 20, "%.4f", h_O[i * N4 + j]);
            if (j == (N - 1)) *ptr++ = '\n';
            else *ptr++ = ' ';

        }
    }

    if (ptr > buffer) {
        fwrite(buffer, 1, ptr - buffer, stdout);
    }


    free(h_O);
    free(h_Q);
    free(h_K);
    free(h_V);

    cudaFree(d_O);
    cudaFree(d_Q);
    cudaFree(d_V);
    cudaFree(d_K);
    cudaFree(d_l);
    cudaFree(d_m);


    return 0;
}