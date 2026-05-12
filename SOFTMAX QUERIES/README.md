# Softmax por Queries com CUDA

Implementação GPU em CUDA do cálculo de **Softmax por faixas de colunas** sobre linhas de uma matriz, respondendo a um volume arbitrário de consultas (queries) dentro do limite de **300 segundos**. Toda a computação usa precisão **FP32**.

Para cada query `(L, C1, C2)`, a saída é:

```
softmax(matriz[L][C1 .. C2])
```

---

## Requisitos

- NVIDIA GPU com suporte a CUDA (Compute Capability 7.0+)
- CUDA Toolkit 11.x ou superior
- Compilador `nvcc`

---

## Compilação

```bash
nvcc -O2 -o qsoftmax qsoftmax.cu
```

---

## Execução

O programa lê os dados via **entrada padrão (stdin)** no seguinte formato:

```
M N
matriz[0][0] ... matriz[0][N-1]
...                               ← M linhas da matriz
L C1 C2
L C1 C2
...                               ← queries (quantidade indeterminada)
```

| Parâmetro | Descrição                              | Restrição        |
|-----------|----------------------------------------|------------------|
| `M`       | Número de linhas da matriz             | 2 ≤ M ≤ 4096     |
| `N`       | Número de colunas                      | 2 ≤ N ≤ 4096     |
| Valores   | Elementos da matriz em FP32            | 0.00 ≤ X ≤ 1.00  |
| `L`       | Linha da query                         | 0 ≤ L < M        |
| `C1`      | Coluna inicial da faixa (inclusivo)    | 0 ≤ C1 < N       |
| `C2`      | Coluna final da faixa (inclusivo)      | 0 ≤ C2 < N       |

### Exemplo de uso com arquivo

```bash
./qsoftmax < entrada.txt
```

### Exemplo de entrada

```
4 8
0.34 0.76 0.32 0.02 0.81 0.65 0.59 0.45
0.18 0.63 0.18 0.73 0.10 0.57 0.88 0.68
0.60 0.20 0.11 0.26 0.70 0.54 0.86 0.93
0.35 0.89 0.66 0.02 0.18 0.56 0.79 0.12
0 0 7
2 4 7
3 1 5
2 0 5
0 2 3
```

### Exemplo de saída

```
0.1043 0.1587 0.1022 0.0757 0.1668 0.1421 0.1339 0.1164
0.2334 0.1989 0.2739 0.2938
0.2921 0.2320 0.1224 0.1436 0.2100
0.1983 0.1330 0.1215 0.1412 0.2192 0.1868
0.5744 0.4256
```

Tolerância aceita: variação de até **±0.02** por elemento em relação à referência. Nenhum espaço no final das linhas.

---

## Detalhes de implementação

### Parâmetros de configuração (`#define`)

| Constante    | Valor | Descrição |
|--------------|-------|-----------|
| `BlockSIZE`  | 128   | Threads por bloco nos kernels de scan |
| `BATCHtype`  | 30    | (reservado para tipagem de batch) |
| `strideB`    | 16    | Stride do array de descritores por linha |
| `colsh`      | 32    | Colunas da memória compartilhada de redução |
| `rowsh`      | 8     | Linhas da memória compartilhada de redução |
| `PADDING`    | 1     | Padding para evitar bank conflicts na shmem |
| `MAX_M`      | 4096  | Dimensão máxima de M |
| `MASK`       | 0xffffffff | Máscara para warp shuffle |

### Estruturas de dados

- **`queries`** — armazena `(l, c1, c2, id)` com `uint16_t` para economizar memória; `id` identifica a ordem original da query no batch.
- **`lowqueries`** — versão sem o campo `l`, usada após o agrupamento por linha.
- **`descriptor`** — par `(value, ok)` com `volatile bool` para sincronização lock-free entre blocos no scan distribuído.

### Pipeline de kernels

O processamento é organizado em **3 CUDA streams** para sobrepor leitura de queries, execução de kernels e cópia de resultados:

```
[Stream 0]  MemcpyH2D → sortKernel → redScan → idxKernel → queriesKernel → MemcpyD2H
[Stream 1]  MemcpyH2D → sortKernel → redScan → idxKernel → queriesKernel → MemcpyD2H
[Stream 2]  MemcpyH2D → sortKernel → redScan → idxKernel → queriesKernel → MemcpyD2H
```

### Pré-processamento da matriz (uma vez, antes das queries)

**`redScanExp`** — Prefix sum exclusivo sobre `exp(x)` por linha:
- Lê pares de floats com `float2` para maior largura de banda.
- Aplica `__expf` e armazena a matriz de exponenciais (`d_exp`).
- Executa um **Blelloch scan** (up-sweep + down-sweep) em memória compartilhada 2D (`sh[rowsh][colsh+PADDING]`).
- Sincroniza blocos da mesma linha via descritores (`descriptor`): cada bloco escreve sua soma parcial e sinaliza `ok = true` com `__threadfence()`; blocos posteriores esperam em busy-wait sobre `dsr[...].ok`.
- Produz `d_prefsum`: prefix sum de `exp(a[i][j])` por linha, usado para responder queries em O(1).

### Processamento de queries por batch

Cada batch de até **25.000 queries** passa pelos seguintes kernels:

**`sortKernel`** — Agrupa queries por linha usando histograma atômico (`atomicAdd`), armazenando as queries reorganizadas em `d_lq`.

**`redScan`** — Prefix sum (versão inteira) sobre o histograma para calcular os offsets de cada linha em `d_lq`.

**`idxKernel`** — Reconstrói o array `d_q` ordenado por linha usando os offsets do prefix sum.

**`queriesKernel`** — Para cada query `(L, C1, C2)`:
- Recupera `prefsum[L][C2]` e `prefsum[L][C1-1]` para calcular a soma `sum = exp(C1..C2)` em O(1).
- Divide cada `exp(a[L][j])` por `sum` para obter o softmax da faixa.
- Threads do bloco cooperam com stride `blockDim.x` para cobrir a faixa `[C1, C2]`.

### Saída com buffer manual

Impressão via buffer de 16 MB com `fwrite` para evitar overhead de `printf` com grandes volumes de saída.

---
