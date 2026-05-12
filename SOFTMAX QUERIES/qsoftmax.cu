#include <iostream>
#include <cmath>
#include <mma.h>
#include <algorithm>
#include <cstdint>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// constantes

#define MAX_M 4096
#define BlockSIZE 128
#define BATCHtype 30

#define strideB 16
#define PADDING 1

#define colsh 32
#define rowsh 8

#define MASK 0xffffffff


// uint16_t em vez de int para reduzir o tamanho da struct de 16 para 10 bytes
// importante para queries porque sao alocadas em pinned memory (cudaMallocHost)
// e transferidas em volume alto por batch
struct queries {
    uint16_t l, c1, c2;
    int id; // id para preservar a ordem original
};


// volatile bool é necessario para que o compilador nao otimize o busy-wait em redScanExp/redScan 
// sem volatile o loop "while (!ok)" poderia ser transformado em leitura unica cacheada travando indefinidamente
struct descriptor {
    volatile bool ok = false;
    float value = 0.0f;
};

// versao compacta sem o campo 'l' usado apos o agrupamento por linha no sortkernel
// economiza memoria
struct lowqueries {
    uint16_t c1, c2;
    int id;
};

__global__ void redScanExp(float* a, float* out, float* expmatrix, descriptor* dsr, int* counter, int M, int N) {

    __shared__ int bidx;
    __shared__ float cur_sum;
    // com padding para evitar bankconflict
    __shared__ float sh[rowsh][colsh + PADDING];

    // cada bloco tem um indice unico
    // necessario para o lookback funcionar

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        bidx = atomicAdd(counter, 1);
        cur_sum = 0.0f;
    }

    __syncthreads();

    int blockIdx = bidx;

    __syncthreads();

    int blocksPerLine = (N + BlockSIZE * 2 - 1) / (BlockSIZE * 2);

    // mapeia o indice global

    int row = blockIdx / blocksPerLine;
    int blockIDx = blockIdx % blocksPerLine;

    //int tid = BlockSIZE * 2 * blockIDx + threadIdx.x;
    int tx = threadIdx.x;

    //int txrow = tx / 32;
    //int txcol = tx % 32;
    // 
    //stride alinhado a multiplo de 2 para viabilizar float2.
    int N2 = ((N + 1) / 2) * 2;
    int tx2 = tx * 2;

    int stride2 = (BlockSIZE)*blockIDx;

    // carrega e armazena 2 floats em uma instruco
    float2* a2 = reinterpret_cast<float2*>(a);
    float2* exp2 = reinterpret_cast<float2*>(expmatrix);

    float2 f2 = { 0.0f, 0.0f };


    if ((stride2 + tx) < N / 2) {
        f2 = a2[row * N2 / 2 + stride2 + tx];
        f2.x = __expf(f2.x);
        f2.y = __expf(f2.y);

        exp2[row * N2 / 2 + stride2 + tx] = f2;

    }
    else if ((stride2 + tx) >= N / 2 && ((N + 1) / 2) > (stride2 + tx)) { // caso de borda precisa ser tratado exclusivamente

        f2 = a2[row * N2 / 2 + stride2 + tx];
        f2.x = __expf(f2.x);

        exp2[row * N2 / 2 + stride2 + tx] = f2;
    }
    // deposita os elementos na shmem
    sh[tx2 / 32][(tx2 % 32)] = f2.x;
    sh[tx2 / 32][((tx2 + 1) % 32)] = f2.y;
    __syncthreads();

    // parte de upsweep, constroi a arvore de soma parcial
    // algoritmo de Blelloch

    for (int stride = 1; stride < BlockSIZE * 2; stride <<= 1) {
        int shid = (tx + 1) * (stride * 2) - 1;

        if (shid < BlockSIZE * 2) {
            int shidrow = shid / 32;
            int shidcol = shid % 32;

            int shstriderow = (shid - stride) / 32;
            int shstridecol = (shid - stride) % 32;

            sh[shidrow][shidcol] += sh[shstriderow][shstridecol];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        // coloca a soma total do bloco no descriptor antes de zerar a raiz.
        // __threadfence() garante que 'value' seja visivel para outros blocos
        // antes de setar 'ok = true' para evitar leitura de valor inconsistente
        // no busy-wait dos blocos
        //

        dsr[row * strideB + blockIDx].value = sh[rowsh - 1][colsh - 1];
        __threadfence();
        dsr[row * strideB + blockIDx].ok = true;

        // zera a raiz para iniciar o downsweep, scan exclusivo
        sh[rowsh - 1][colsh - 1] = 0.0f;
    }

    // parte do downsweep
    // propaga os prefixos pela arvore construida
    // transforma a arvore de reduçao em prefix sum exclusivo

    for (int stride = BlockSIZE; stride > 0; stride >>= 1) {
        int shid = (tx + 1) * (stride * 2) - 1;

        if (shid < BlockSIZE * 2) {
            int shidrow = shid / 32;
            int shidcol = shid % 32;

            int shstriderow = (shid - stride) / 32;
            int shstridecol = (shid - stride) % 32;

            float right = sh[shidrow][shidcol];
            float left = sh[shstriderow][shstridecol];

            sh[shstriderow][shstridecol] = right;
            sh[shidrow][shidcol] = right + left;
        }

        __syncthreads();
    }

    float thispsum = 0.0f;

    // busy-wait, espera blocos anteriores da mesma linha publicarem suas somas
    // Cada thread tx le o descriptor do bloco tx (< blockIDx)
    // acumulando a soma de todos os blocos que vieram antes na linha
    // implementaçao do lookback (ele olha para tras nos blocos anteriores)

    if (threadIdx.y == 0 && tx < blockIDx) {
        while (!dsr[row * strideB + tx].ok) {}

        thispsum = dsr[row * strideB + tx].value;

    }

    // reduçao intra warp

    if (threadIdx.y == 0 && threadIdx.x < 32) {
        for (int i = 16; i > 0; i >>= 1) {
            thispsum += __shfl_down_sync(MASK, thispsum, i, 32);
        }

        if (threadIdx.x == 0) {
            cur_sum = thispsum;
        }
    }
    __syncthreads();

    float accumpsum = cur_sum;

    __syncthreads();


    float2* out2 = reinterpret_cast<float2*>(out);

    // escreve o prefix sum final

    if (2 * (stride2 + tx) < N2) {
        float2 f2x = {};
        f2x.x = sh[tx2 / 32][(tx2 % 32)] + accumpsum + f2.x;
        f2x.y = sh[tx2 / 32][(tx2 % 32) + 1] + accumpsum + f2.y;

        out2[row * N2 / 2 + stride2 + tx] = f2x;
    }


}

// reinicializa descritores e counter antes de cada batch de queries
// necessario porque os descritores sao reutilizados entre batches
// e o counter precisa começar do zero para o atomicAdd funcionar corretamente

__global__ void initDescriptorCounter(descriptor* dsr, int M, int* counter) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row == 0 && threadIdx.x == 0) *counter = 0;

    if (row < M) {
        dsr[row * strideB + threadIdx.x].ok = false;
        dsr[row * strideB + threadIdx.x].value = 0.0f;
    }
}


// Agrupa queries por linha antes de executar o softmax.
// ideia: queries da mesma linha acessam a mesma regiao de memoria
// agrupando, maximizamos reuso de cache e evitamos saltar entre linhas a cada query.
// para entradas grandes faz diferença
// Usa atomicAdd no histograma para insercao em "niveis" sem conflito
__global__ void sortKernel(queries* q, int* historiograma, int prof, lowqueries* lq) {

    int tid = threadIdx.x + (blockDim.x * blockIdx.x);

    if (tid >= prof) return;

    if (tid < prof) {
        int value = q[tid].l;
        int pos;
        pos = atomicAdd(&historiograma[value], 1);

        lq[value * prof + pos].c1 = q[tid].c1;
        lq[value * prof + pos].c2 = q[tid].c2;
        lq[value * prof + pos].id = q[tid].id;
    }
    //printf("%d", tid);
}

// initHist, zera o histograma antes de cada batch
// Loop com stride blockDim.x para cobrir MAX_M entradas com um unico bloco

__global__ void initHist(int* historiograma) {

    int tx = threadIdx.x;

    for (int i = 0; i < MAX_M; i += blockDim.x) historiograma[tx + i] = 0;

}

// redScan, prefix sum exclusivo sobre o histograma de inteiros.
// Mesmo algoritmo de Blelloch de redScanExp mas operando em int2.
// Resultado em 'out': out[i] = numero total de queries nas linhas 0..i-1
// usado como offset base para reconstruir d_q ordenado em idxKernel

__global__ void redScan(int* a, int* out, descriptor* dsr, int* counter, int M, int N) {

    __shared__ int bidx;
    __shared__ int cur_sum;
    __shared__ int sh[rowsh][colsh + PADDING];

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        bidx = atomicAdd(counter, 1);
        cur_sum = 0;
    }

    __syncthreads();

    int blockIdx = bidx;
    //float
    __syncthreads();

    int blocksPerLine = (N + BlockSIZE * 2 - 1) / (BlockSIZE * 2);

    int row = blockIdx / blocksPerLine;
    int blockIDx = blockIdx % blocksPerLine;

    int tx = threadIdx.x;

    int N2 = ((N + 1) / 2) * 2;
    int tx2 = tx * 2;

    int stride2 = (BlockSIZE)*blockIDx;

    int2* a2 = reinterpret_cast<int2*>(a);

    int2 f2 = { 0, 0 };

    if ((stride2 + tx) < N / 2) {
        f2 = a2[row * N2 / 2 + stride2 + tx];

    }
    else if ((stride2 + tx) >= N / 2 && ((N + 1) / 2) > (stride2 + tx)) {

        f2 = a2[row * N2 / 2 + stride2 + tx];

    }

    sh[tx2 / 32][(tx2 % 32)] = f2.x;
    sh[tx2 / 32][((tx2 + 1) % 32)] = f2.y;
    __syncthreads();

    for (int stride = 1; stride < BlockSIZE * 2; stride <<= 1) {
        int shid = (tx + 1) * (stride * 2) - 1;

        if (shid < BlockSIZE * 2) {
            int shidrow = shid / 32;
            int shidcol = shid % 32;

            int shstriderow = (shid - stride) / 32;
            int shstridecol = (shid - stride) % 32;

            sh[shidrow][shidcol] += sh[shstriderow][shstridecol];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        //b_sum[row * strideB + blockIdx.x] = sh[BlockSIZE * 2 - 1];

        dsr[row * strideB + blockIDx].value = sh[rowsh - 1][colsh - 1];
        __threadfence();
        dsr[row * strideB + blockIDx].ok = true;

        sh[rowsh - 1][colsh - 1] = 0;
    }

    for (int stride = BlockSIZE; stride > 0; stride >>= 1) {
        int shid = (tx + 1) * (stride * 2) - 1;

        if (shid < BlockSIZE * 2) {
            int shidrow = shid / 32;
            int shidcol = shid % 32;

            int shstriderow = (shid - stride) / 32;
            int shstridecol = (shid - stride) % 32;

            int right = sh[shidrow][shidcol];
            int left = sh[shstriderow][shstridecol];

            sh[shstriderow][shstridecol] = right;
            sh[shidrow][shidcol] = right + left;
        }

        __syncthreads();
    }

    int thispsum = 0;

    if (threadIdx.y == 0 && tx < blockIDx) {
        while (!dsr[row * strideB + tx].ok) {}

        thispsum = dsr[row * strideB + tx].value;

    }

    if (threadIdx.y == 0 && threadIdx.x < 32) {
        for (int i = 16; i > 0; i >>= 1) {
            thispsum += __shfl_down_sync(MASK, thispsum, i, 32);
        }

        if (threadIdx.x == 0) {
            cur_sum = thispsum;
        }
    }
    __syncthreads();

    int accumpsum = cur_sum;

    __syncthreads();


    int2* out2 = reinterpret_cast<int2*>(out);

    if (2 * (stride2 + tx) < N2) {
        int2 f2x = {};
        f2x.x = sh[tx2 / 32][(tx2 % 32)] + accumpsum + f2.x;
        f2x.y = sh[tx2 / 32][(tx2 % 32) + 1] + accumpsum + f2.y;

        out2[row * N2 / 2 + stride2 + tx] = f2x;
    }


}

// queriesKernel responde cada query em O(1) usando o prefix sum precomputado
// Para softmax(a[l][c1..c2]):
// sum = prefsum[l][c2] - prefsum[l][c1-1]  (soma de exp na faixa)
// resultado[j] = exp(a[l][j]) / sum
// Evita recalcular exp() ou iterar a linha inteira por query.

__global__ void queriesKernel(float* a, float* b, float* out, queries* q, int M, int N, int N2, int BATCH) {
    // out = array de saida;
    // a = array com psum;
    // b = array com expf;
    // N2 = ((N+1)/2) * 2; faz-se assim para uso do float2
    // os indices c1, c2, e l sao 0-indexed

    int rowq = blockIdx.y * blockDim.y + threadIdx.y;

    if (rowq >= BATCH) return;

    int l = q[rowq].l;
    int c1 = q[rowq].c1;
    int c2 = q[rowq].c2;
    int id = q[rowq].id;


    // c1-- converte para o indice do prefix sum exclusivo
    // prefsum[c1-1] e a soma acumulada antes de c1
    c1--;

    float initpsum = 0.0f;
    float endpsum = 0.0f;
    // c1 < 0 significa query comecando na coluna 0: prefixo inicial = 0.
    if (c1 >= 0) initpsum = a[l * N2 + c1];
    endpsum = a[l * N2 + c2];


    // div = soma de exp na faixa [c1+1, c2] = denominador do softmax.
    float div = endpsum - initpsum;
    if (div == 0.0f) div = 1.0f;

    // Threads do bloco cooperam com stride blockDim.x para cobrir a faixa
     // permitindo queries longas paralelas
    for (int i = c1 + 1; i <= (c2 + blockDim.x - 1); i += blockDim.x) {
        if ((threadIdx.x + i) <= c2) {

            float ans = b[l * N2 + (threadIdx.x + i)];
            ans = ((float)ans / (float)div);

            //if (id == 0) printf("%d ", (threadIdx.x + i));
            out[id * N2 + (threadIdx.x + i) - (c1 + 1)] = ans;
        }
    }

}

// idxKernel reconstroi d_q ordenado por linha usando os offsets do prefix sum
// Inverte o agrupamento feito por sortKernel
// para cada linha tid de d_lq para as posicoes corretas em d_q
// usando o offset calculado por redScan mantendo o id original para a saida

__global__ void idxKernel(queries* q, int* historiograma, int* offset, int prof, lowqueries* lq, int M) {
    int tx = threadIdx.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int limit;

    //if (tid >= M) return;

    limit = historiograma[tid]; // numero de queries na linha tid

    int offsetTX;

    // offset[tid-1] = posicao inicial das queries da linha tid no array d_q ordenado.
    if (tid == 0) offsetTX = 0;
    else offsetTX = offset[tid - 1];

    for (int i = 0; i < (limit + blockDim.y - 1); i += blockDim.y) {
        if (i + threadIdx.y < limit) {
            // tid aqui representa a linha 0 a 4095

            //if(prof + i + threadIdx.y >= M * 10000)printf("%d > %d\n", M * 10000, tid * prof + i + threadIdx.y);
            //if(offsetTX + i + threadIdx.y >= 10000)printf("%d \n", offsetTX + i + threadIdx.y);

            q[offsetTX + i + threadIdx.y].l = tid;
            q[offsetTX + i + threadIdx.y].c1 = lq[tid * prof + i + threadIdx.y].c1;
            q[offsetTX + i + threadIdx.y].c2 = lq[tid * prof + i + threadIdx.y].c2;
            q[offsetTX + i + threadIdx.y].id = lq[tid * prof + i + threadIdx.y].id;

            //if(q[offsetTX + i + threadIdx.y].id == 0) printf("%d \n", q[offsetTX + i + threadIdx.y].id);
        }
    }
    //printf("%d %d\n", tid, threadIdx.y);
}

int main() {
    int M, N, N2, BATCH;
    // 3 streams: permite sobrepor leitura de queries (CPU), execucao de kernels
    // e copia de resultados (D2H) em paralelo, escondendo latencia de I/O.
    const int num_streams = 3;
    BATCH = 25000;

    scanf("%d%d", &M, &N);

    N2 = ((N + 1) / 2) * 2;

    size_t bytes_matrix = N2 * M * sizeof(float);
    size_t bytes_descriptor = M * strideB * sizeof(descriptor);
    size_t bytes_queries = BATCH * sizeof(queries);
    size_t bytes_out = BATCH * N2 * sizeof(float);
    size_t buffer_size = 16 * 1024 * 1024;

    // Inicialização da matrix das queries

    int* historiograma_f, * historiograma_prefsum;

    float* h_init, * h_out[num_streams];
    float* d_init, * d_prefsum, * d_exp, * d_out[num_streams];

    descriptor* d_dsr;

    int* counter;

    queries* d_q[num_streams], * h_q[num_streams];

    lowqueries* d_lq;

    cudaStream_t streams_out[num_streams];
    cudaStream_t streams_querie[num_streams];
    cudaStream_t kernel_done[num_streams];

    char* buffer;

    h_init = (float*)malloc(bytes_matrix);
    buffer = (char*)malloc(buffer_size);

    int cnt_stream[num_streams];

    // cudaMallocHost (pinned memory) para h_q e h_out
    // permite cudaMemcpyAsync entre host e device
    // Com malloc normal a copia seria sincronizada internamente pelo driver, anulando o beneficio dos streams

    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams_querie[i]);
        cudaStreamCreate(&streams_out[i]);
        cudaMallocHost(&h_q[i], bytes_queries);
        cudaMallocHost(&h_out[i], bytes_out);
        cudaMalloc(&d_out[i], bytes_out);
        cudaMalloc(&d_q[i], bytes_queries);

        cnt_stream[i] = 0;
    }

    //h_q = (queries*)malloc(bytes_queries);
    //h_out = (float*)malloc(bytes_out);

    // Matrizes iniciais
    cudaMalloc(&d_init, bytes_matrix);
    cudaMalloc(&d_prefsum, bytes_matrix);
    cudaMalloc(&d_exp, bytes_matrix);

    //Descriptor
    cudaMalloc(&d_dsr, bytes_descriptor);

    //Queries e Auxiliares
    cudaMalloc(&counter, sizeof(int));
    cudaMalloc(&historiograma_f, 4096 * sizeof(int));
    cudaMalloc(&historiograma_prefsum, 4096 * sizeof(int));
    cudaMalloc(&d_lq, 4097 * BATCH * sizeof(lowqueries));

    char* ptr = buffer;

    //Ler matriz
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) scanf("%f", &h_init[i * N2 + j]);
    }

    cudaMemcpy(d_init, h_init, bytes_matrix, cudaMemcpyHostToDevice);

    dim3 blockSizePsum(BlockSIZE, 1);
    dim3 blockCntPsum((N + BlockSIZE * 2 - 1) / (BlockSIZE * 2), M);

    dim3 blockSizeInitDsrCnt(16, 32);
    dim3 blockCntInitDsrCnt(1, (M + blockSizeInitDsrCnt.y - 1) / blockSizeInitDsrCnt.y);

    initDescriptorCounter << <blockCntInitDsrCnt, blockSizeInitDsrCnt >> > (d_dsr, M, counter);

    cudaDeviceSynchronize();

    // pre-processa a matriz uma unica vez calcula exp e prefix sum de exp por linha.
    redScanExp << <blockCntPsum, blockSizePsum >> > (d_init, d_prefsum, d_exp, d_dsr, counter, M, N);

    int cnt_q = 0;
    bool isOk = true;
    int stream_read = 0; // stream que esta lendo/processando o batch atual
    int stream_write = 0; // stream cujo resultado esta pronto para impressao

    // abordagem consumer/producer
    // enche os primeiros (num_streams - 1) slots antes de entrar no loop principal.
    // Garante que sempre haja um batch pronto para imprimir enquanto o proximo e processado.
    // prologo

    for (int j = 0; j < num_streams - 1; j++) {
        cnt_q = 0;

        for (int i = 0; i < BATCH; i++) {

            if (scanf("%hu%hu%hu", &h_q[stream_read][i].l, &h_q[stream_read][i].c1, &h_q[stream_read][i].c2) == 3) {
                cnt_q++;
                h_q[stream_read][i].id = i;
            }
            else {
                isOk = false;
                break;
            }

        }
        if (cnt_q == 0) {
            break;
        }

        cnt_stream[stream_read] = cnt_q;
        cudaMemcpyAsync(d_q[stream_read], h_q[stream_read], cnt_stream[stream_read] * sizeof(queries), cudaMemcpyHostToDevice, streams_querie[stream_read]);

        initHist << <1, BlockSIZE, 0, streams_querie[stream_read] >> > (historiograma_f);

        initDescriptorCounter << <blockCntInitDsrCnt, blockSizeInitDsrCnt, 0, streams_querie[stream_read] >> > (d_dsr, M, counter);

        dim3 blockSizeSort(512, 1);
        dim3 blockCntSort((cnt_stream[stream_read] + 511) / 512, 1);


        //cudaStreamSynchronize(streams_querie[stream_read]);
        // Pipeline por batch: sort -> prefix sum do histograma -> reordenacao -> softmax por query
        sortKernel << <blockCntSort, blockSizeSort, 0, streams_querie[stream_read] >> > (d_q[stream_read], historiograma_f, cnt_stream[stream_read], d_lq);

        dim3 blockSizeRedScan(BlockSIZE, 1);
        dim3 blockCntRedScan(4096 / BlockSIZE, 1);

        redScan << <blockCntRedScan, blockSizeRedScan, 0, streams_querie[stream_read] >> > (historiograma_f, historiograma_prefsum, d_dsr, counter, 1, 4096);

        dim3 blockSizeIdx(16, 32);
        dim3 blockCntIdx((M + blockSizeIdx.x - 1) / blockSizeIdx.x, 1);


        idxKernel << <blockCntIdx, blockSizeIdx, 0, streams_querie[stream_read] >> > (d_q[stream_read], historiograma_f, historiograma_prefsum, cnt_stream[stream_read], d_lq, M);


        dim3 blockSizeQuerie(32, 2);
        dim3 blockCntQuerie(1, (cnt_q + 1) / 2);

        queriesKernel << <blockCntQuerie, blockSizeQuerie, 0, streams_querie[stream_read] >> > (d_prefsum, d_exp, d_out[stream_read], d_q[stream_read], M, N, N2, cnt_stream[stream_read]);

        cudaStreamSynchronize(streams_querie[stream_read]);

        cudaMemcpyAsync(h_out[stream_read], d_out[stream_read], cnt_stream[stream_read] * N2 * sizeof(float), cudaMemcpyDeviceToHost, streams_out[stream_read]);

        stream_read = (stream_read + 1) % num_streams;
    }
    // imprime stream_write enquanto processa stream_read em paralelo.
    while (isOk) {
        cnt_q = 0;

        // Espera a copia D2H do batch stream_write terminar antes de imprimir.
        cudaStreamSynchronize(streams_out[stream_write]);

        for (int i = 0; i < cnt_stream[stream_write]; i++) {
            int c1, c2;
            c1 = h_q[stream_write][i].c1;
            c2 = h_q[stream_write][i].c2;

            for (int j = c1; j <= c2; j++) {

                if (ptr + 25 > buffer + buffer_size) {
                    fwrite(buffer, 1, ptr - buffer, stdout);
                    ptr = buffer;
                }

                ptr += snprintf(ptr, 20, "%.4f", h_out[stream_write][i * N2 + (j - c1)]);
                if (j == c2) *ptr++ = '\n';
                else *ptr++ = ' ';

            }
        }

        if (ptr > buffer) {
            fwrite(buffer, 1, ptr - buffer, stdout);
        }

        cnt_stream[stream_write] = 0;

        stream_write = (stream_write + 1) % num_streams;

        for (int i = 0; i < BATCH; i++) {

            if (scanf("%hu%hu%hu", &h_q[stream_read][i].l, &h_q[stream_read][i].c1, &h_q[stream_read][i].c2) == 3) {
                cnt_q++;
                h_q[stream_read][i].id = i;
            }
            else break;

        }

        if (cnt_q == 0) break;

        cnt_stream[stream_read] = cnt_q;

        cudaMemcpyAsync(d_q[stream_read], h_q[stream_read], cnt_stream[stream_read] * sizeof(queries), cudaMemcpyHostToDevice, streams_querie[stream_read]);

        initHist << <1, BlockSIZE, 0, streams_querie[stream_read] >> > (historiograma_f);

        initDescriptorCounter << <blockCntInitDsrCnt, blockSizeInitDsrCnt, 0, streams_querie[stream_read] >> > (d_dsr, M, counter);

        dim3 blockSizeSort(512, 1);
        dim3 blockCntSort((cnt_q + 511) / 512, 1);

        sortKernel << <blockCntSort, blockSizeSort, 0, streams_querie[stream_read] >> > (d_q[stream_read], historiograma_f, cnt_stream[stream_read], d_lq);

        dim3 blockSizeRedScan(BlockSIZE, 1);
        dim3 blockCntRedScan(4096 / BlockSIZE, 1);

        redScan << <blockCntRedScan, blockSizeRedScan, 0, streams_querie[stream_read] >> > (historiograma_f, historiograma_prefsum, d_dsr, counter, 1, 4096);


        dim3 blockSizeIdx(16, 32);
        dim3 blockCntIdx((M + blockSizeIdx.x - 1) / blockSizeIdx.x, 1);

        idxKernel << <blockCntIdx, blockSizeIdx, 0, streams_querie[stream_read] >> > (d_q[stream_read], historiograma_f, historiograma_prefsum, cnt_stream[stream_read], d_lq, M);

        dim3 blockSizeQuerie(32, 2);
        dim3 blockCntQuerie(1, (cnt_q + 1) / 2);

        queriesKernel << <blockCntQuerie, blockSizeQuerie, 0, streams_querie[stream_read] >> > (d_prefsum, d_exp, d_out[stream_read], d_q[stream_read], M, N, N2, cnt_stream[stream_read]);

        cudaStreamSynchronize(streams_querie[stream_read]);

        cudaMemcpyAsync(h_out[stream_read], d_out[stream_read], cnt_stream[stream_read] * N2 * sizeof(float), cudaMemcpyDeviceToHost, streams_out[stream_read]);

        stream_read = (stream_read + 1) % num_streams;

    }
    // Drena os batches restantes que ja estao em voo nos streams
    // epilogo
    while (cnt_stream[stream_write] > 0) {

        cudaStreamSynchronize(streams_out[stream_write]);

        for (int i = 0; i < cnt_stream[stream_write]; i++) {
            int c1, c2;
            c1 = h_q[stream_write][i].c1;
            c2 = h_q[stream_write][i].c2;

            for (int j = c1; j <= c2; j++) {

                if (ptr + 25 > buffer + buffer_size) {
                    fwrite(buffer, 1, ptr - buffer, stdout);
                    ptr = buffer;
                }

                ptr += snprintf(ptr, 20, "%.4f", h_out[stream_write][i * N2 + (j - c1)]);
                if (j == c2) *ptr++ = '\n';
                else *ptr++ = ' ';

            }
        }

        if (ptr > buffer) {
            fwrite(buffer, 1, ptr - buffer, stdout);
        }

        cnt_stream[stream_write] = 0;
        stream_write = (stream_write + 1) % num_streams;
    }

    // usar free no stream etc
    ;
    for (int i = 0; i < num_streams; i++) {

        cudaStreamDestroy(streams_querie[i]);
        cudaStreamDestroy(streams_out[i]);

        cudaFreeHost(h_q[i]);
        cudaFreeHost(h_out[i]);
        cudaFree(d_out[i]);
        cudaFree(d_q[i]);

    }

    free(h_init);

    cudaFree(d_init);
    cudaFree(d_prefsum);
    cudaFree(d_exp);

    cudaFree(d_dsr);

    cudaFree(counter);
    cudaFree(historiograma_f);
    cudaFree(historiograma_prefsum);
    cudaFree(d_lq);

    return 0;
}