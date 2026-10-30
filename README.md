# CUDA Transformer Kernels

Implementações em CUDA C++ de multiplicação de matrizes, atenção, softmax por faixas e seleção top-k. O projeto foi desenvolvido individualmente para os desafios de GPU da [Neospace AI](https://www.instagram.com/neospace_ai), como estudo das operações que aparecem em redes neurais e Transformers.

As operações principais usam FP32 na GPU. A CPU lê as entradas, organiza as transferências e imprime os resultados. Cada desafio é um programa independente, com seu próprio `main`, formato de entrada e documentação.

## Operações implementadas

| Desafio | Operação | Estratégia principal | Código e instruções |
|---|---|---|---|
| B | Multiplicação de matrizes quadradas | Tiling em memória compartilhada, padding e acumulação com `fmaf` | [GEMM](https://github.com/ItaloLRC/cuda-transformer-kernels/tree/main/GEMM), `matriz.cu` |
| C | Atenção scaled dot-product | Tiling, vetorização com `float4`, reduções por warp e normalização incremental | [FLASHATTENTION](https://github.com/ItaloLRC/cuda-transformer-kernels/tree/main/FLASHATTENTION), `attention.cu` |
| D | Softmax para consultas sobre intervalos de colunas | Prefix sums, agrupamento por linha e processamento em lotes | [SOFTMAX QUERIES](https://github.com/ItaloLRC/cuda-transformer-kernels/tree/main/SOFTMAX%20QUERIES), `qsoftmax.cu` |
| E | Seleção dos maiores valores de cada linha | Ordenação bitônica parcial e redução do conjunto de candidatos | [TOPK](https://github.com/ItaloLRC/cuda-transformer-kernels/tree/main/TOPK), `topk.cu` |

### Multiplicação de matrizes

O programa calcula C = A × B para duas matrizes N × N. Blocos de threads carregam trechos de A e B na memória compartilhada e reutilizam esses valores para calcular os produtos. O kernel trabalha com tiles de 16 × 16 e agrupa oito trechos ao longo da dimensão de redução.

### Atenção

O programa recebe Q, K e V com M linhas e N colunas e calcula:

```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(dk)) * V
```

N corresponde a dk. A implementação acompanha o máximo e a soma de exponenciais durante o processamento dos tiles, uma estratégia inspirada no FlashAttention. Ela evita armazenar a matriz completa de atenção M × M. As larguras de memória são arredondadas para múltiplos de quatro para os acessos com `float4`.

### Softmax por consultas

Cada consulta escolhe uma linha e uma faixa inclusiva de colunas. O pré-processamento calcula exponenciais e prefix sums por linha. A soma usada para normalizar uma faixa é obtida em O(1); produzir seus valores de saída continua custando O(k), onde k é o tamanho da faixa.

As consultas são agrupadas por linha e processadas em lotes de até 25.000. O código mantém três posições de lote em circulação, com streams separados para cálculo e cópia dos resultados. O scan entre blocos usa descritores e sinalização de memória; o tema é discutido na referência de prefix scan abaixo.

### Top-k por linha

K é um percentual entre 5 e 15. Para uma linha com N elementos, o programa imprime os `ceil(N*K/100)` maiores valores em ordem decrescente. A seleção usa ordenação bitônica de um conjunto parcial e sucessivas reduções dos candidatos. M e N devem ser potências de dois, conforme as restrições do desafio.

## Compilar e executar

É necessário ter uma GPU NVIDIA, driver e CUDA Toolkit compatíveis, `nvcc` e um compilador host suportado pelo Toolkit. As instruções dos desafios usam Compute Capability 7.0 ou superior; a arquitetura selecionada na compilação deve corresponder à GPU de execução.

Na raiz do repositório:

```bash
nvcc -O2 "GEMM/matriz.cu" -o matriz
nvcc -O2 "FLASHATTENTION/attention.cu" -o attention
nvcc -O2 "SOFTMAX QUERIES/qsoftmax.cu" -o qsoftmax
nvcc -O2 "TOPK/topk.cu" -o topk
```

Os executáveis leem de stdin e escrevem em stdout. Em Bash, por exemplo:

```bash
./matriz < entrada.txt > saida.txt
```

| Programa | Entrada |
|---|---|
| `matriz` | N, seguido de duas matrizes N × N |
| `attention` | M e N, seguidos das matrizes Q, K e V, cada uma M × N |
| `qsoftmax` | M e N, a matriz M × N e consultas `L C1 C2` até o fim da entrada |
| `topk` | K, M e N, seguidos da matriz M × N |

As subpastas contêm exemplos completos e as restrições de tamanho, precisão e formato de saída. Atenção, consultas e top-k trabalham com dimensões de até 4096 nas condições descritas nesses documentos; top-k exige dimensões que sejam potências de dois.

## Técnicas de programação em GPU

- Tiling e reuso de dados em memória compartilhada.
- Acessos vetorizados com `float2` e `float4`.
- Reduções por warp e normalização incremental na atenção.
- Prefix sums e agrupamento de consultas por linha.
- Processamento em lotes e sobreposição de operações com CUDA streams.
- Seleção top-k com ordenação bitônica parcial.

Cada subpasta documenta o formato de entrada, a saída esperada e os parâmetros do respectivo desafio.

## Referências

- Vaswani et al., [Attention is All You Need](https://arxiv.org/pdf/1706.03762), 2017.
- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), 2022.
- Merrill e Garland, [Single-pass Parallel Prefix Scan with Decoupled Look-back](https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf), 2016.
- NVIDIA, [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf).
- [Enunciados dos desafios Neospace AI](https://drive.google.com/drive/folders/1PFsfI1s2XMLAJfMpT6yG5XrmC7A5W4gR).
