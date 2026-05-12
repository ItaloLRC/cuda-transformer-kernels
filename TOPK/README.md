# Top-K por Linha com CUDA

Implementação GPU em CUDA para seleção e ordenação decrescente dos **Top-K maiores valores** em cada linha de uma matriz, dentro do limite de **300 segundos**. Toda a computação roda em GPU; apenas entrada e saída de dados ocorrem na CPU.

O valor de K é interpretado como **percentual** da quantidade de colunas, com arredondamento para cima:

```
P = ceil(N * K / 100)
```

Exemplos:
- K=10, N=32 → P = ceil(3.2) = **4** maiores por linha
- K=15, N=16 → P = ceil(2.4) = **3** maiores por linha

---

## Requisitos

- NVIDIA GPU com suporte a CUDA
- CUDA Toolkit 11.x ou superior
- Compilador `nvcc`
- M e N devem ser **potências de 2**

---

## Compilação

```bash
nvcc -O2 -o topk topk.cu
```

---

## Execução

O programa lê os dados via **entrada padrão (stdin)** no seguinte formato:

```
K M N
matriz[0][0] ... matriz[0][N-1]
...                               ← M linhas da matriz
```

| Parâmetro | Descrição                                      | Restrição              |
|-----------|------------------------------------------------|------------------------|
| `K`       | Percentual dos maiores valores a selecionar    | 5 ≤ K ≤ 15             |
| `M`       | Número de linhas da matriz                     | 2 ≤ M ≤ 4096, pot. de 2|
| `N`       | Número de colunas da matriz                    | 2 ≤ N ≤ 4096, pot. de 2|
| Valores   | Elementos da matriz com 3 casas decimais       | 0.000 ≤ X ≤ 1.000      |

### Exemplo de uso com arquivo

```bash
./topk < entrada.txt
```

### Exemplo de entrada

```
<!-- Insira aqui um exemplo de entrada -->
```

### Exemplo de saída

```
<!-- Insira aqui a saída esperada -->
```

A saída exibe os `P` maiores valores de cada linha em **ordem decrescente**, com 3 casas decimais, sem espaço no final das linhas.

---

## Detalhes de implementação

### Cálculo de P e P2

```
P  = ceil(N * K / 100)   ← quantidade real de elementos a selecionar
P2 = próxima potência de 2 ≥ P  ← tamanho do bitonic sort inicial
```

`P2` é necessário porque o Bitonic Sort exige tamanho potência de 2.

### Parâmetros de configuração (`#define`)

| Constante | Valor | Descrição |
|-----------|-------|-----------|
| `MAXREG`  | 16    | Tamanho máximo de registradores por thread (reservado) |
| `ROWSH`   | 1     | Linhas de memória compartilhada por bloco (reservado) |
| `MAXN`    | 4096  | Dimensão máxima de N |

### Kernel `meu_kernel`

Executa o Top-K in-place diretamente na memória global, em duas fases:

**Fase 1 — Bitonic Sort parcial sobre `P2` elementos:**
- Ordena os primeiros `P2` elementos de cada linha usando **Bitonic Sort** completo.
- Threads do bloco cooperam com stride `blockDim.x` para cobrir todos os índices.
- A direção de ordenação é controlada pelo bit `(dir & i)`: produz ordem decrescente nos segmentos de interesse.
- Ao final, os `P2` primeiros elementos da linha estão ordenados de forma bitônica.

**Fase 2 — Redução iterativa para isolar os P maiores:**
- A cada iteração, compara pares `(idx, idx + P)` com `fmaxf` e mantém apenas o maior, efetivamente eliminando metade dos candidatos por passo.
- Em seguida, compacta o array descartando posições eliminadas: cada elemento `i` copia de `src = i + (i/P)*P`.
- Reaplica um Bitonic Sort parcial (de tamanho `P/2 → P`) sobre o subarray compactado para manter a ordem decrescente.
- Repete até que apenas `P` elementos, nos índices `[0, P)`, contenham os maiores valores em ordem decrescente.

### Saída com buffer manual

Impressão via buffer de 16 MB com `fwrite` para evitar overhead de `printf` por elemento em matrizes grandes.

---

## Limitações conhecidas

- M e N **precisam ser potências de 2** (restrição do enunciado e do algoritmo).
- O kernel opera in-place: a matriz original é modificada na GPU.
- Não há verificação de erros CUDA nas chamadas de alocação e cópia (a macro `CUDA_CHECK` está definida mas não aplicada nas chamadas do `main`).
- `blockDim.x = 256` é fixo; ajustes podem ser necessários para GPUs com menor ocupância.