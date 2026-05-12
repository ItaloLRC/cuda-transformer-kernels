# CUDA Kernels — Desafios Neospace AI

Implementações em CUDA C++ desenvolvidas para os desafios mensais de GPU promovidos pela [Neospace AI](https://www.instagram.com/neospace_ai), avaliados em GPU **NVIDIA GB200**. Todos os kernels rodam inteiramente na GPU (exceto I/O) e usam precisão **FP32**.

Os desafios foram feitos de forma **individual**, e serviram como ponto de entrada para entender como primitivas de Deep Learning funcionam de baixo nível, antes de frameworks como PyTorch.

---

## Desafios

### [Desafio B — Multiplicação de Matrizes](./matriz/)
`matriz.cu` · 40 pts

Multiplicação de matrizes quadradas N×N (até 4096×4096) com tiling em memória compartilhada e vetorização com `float4`. Ponto de partida para entender GEMM na GPU — operação base de qualquer rede neural.

---

### [Desafio C — Attention(Q, K, V)](./attention/)
`attention.cu` · 50 pts

Implementação do mecanismo de atenção scaled dot-product conforme o paper original do Transformer (Vaswani et al., 2017), com estratégia de tiling inspirada no FlashAttention (Dao et al., 2022): evita materializar a matriz M×M na memória compartilhada usando uma recorrência online de máximo e soma de exponenciais.

```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(dk)) * V
```

---

### [Desafio D — Queries de Softmax](./qsoftmax/)
`qsoftmax.cu` · 50 pts · TL: 300s

Softmax por faixas de colunas para volume arbitrário de queries. A chave é pré-computar o prefix sum de `exp(a[i][j])` por linha uma única vez — respondendo cada query em O(1) em vez de varrer a linha inteira. O prefix sum distribuído entre blocos usa o algoritmo **Decoupled Look-back** (Merrill & Garland, 2016), com sincronização lock-free via descriptors e `__threadfence()`. As queries são processadas em batches de 25k com 3 CUDA streams para sobrepor I/O e computação.

---

### [Desafio E — Top-K](./topk/)
`topk.cu` · 50 pts · TL: 300s

Seleção e ordenação decrescente dos K% maiores valores por linha (K entre 5% e 15%). Evita ordenar N inteiro: faz um Bitonic Sort parcial sobre os primeiros P=ceil(N*K/100) elementos e itera reduzindo os candidatos com `fmaxf` + compactação, extraindo o top-K sem varrer o array mais de `log(N/P)` vezes.

---

## Algoritmos e conceitos cobertos

| Conceito | Onde aparece |
|---|---|
| Tiled GEMM com shmem | matriz, attention |
| FlashAttention (recorrência online de softmax) | attention |
| Blelloch Scan (up-sweep / down-sweep) | qsoftmax |
| Decoupled Look-back (prefix sum entre blocos) | qsoftmax |
| Bitonic Sort parcial | topk |
| Vetorização com `float2` / `float4` | attention, qsoftmax |
| Warp shuffle (`__shfl_down_sync`) | attention, qsoftmax |
| CUDA Streams + pinned memory | qsoftmax |
| Bank conflict avoidance (PADDING na shmem) | todos |

---

## Referências

- Vaswani et al., [Attention is All You Need](https://arxiv.org/pdf/1706.03762) (2017)
- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (2022)
- Merrill & Garland, [Single-pass Parallel Prefix Scan with Decoupled Look-back](https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf) (2016)
- NVIDIA, [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf)
- Enunciados originais: [Drive Neospace AI](https://drive.google.com/drive/folders/1PFsfI1s2XMLAJfMpT6yG5XrmC7A5W4gR)
