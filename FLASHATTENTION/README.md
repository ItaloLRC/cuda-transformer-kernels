# Atenção Transformer com CUDA (FlashAttention-style)

Implementação GPU em CUDA do mecanismo de atenção scaled dot-product conforme o artigo  
["Attention is All You Need" (Vaswani et al., 2017)](https://arxiv.org/pdf/1706.03762), com estratégia de tiling inspirada no FlashAttention para reduzir o uso de memória compartilhada. Toda a computação usa precisão **FP32**.

A fórmula implementada é:

```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(dk)) * V
```

onde `dk = N`.

---

## Requisitos

- NVIDIA GPU com suporte a CUDA (Compute Capability 7.0+)
- CUDA Toolkit 11.x ou superior
- Compilador `nvcc`

---

## Compilação

```bash
nvcc -O2 -o attention attention.cu
```

---

## Execução

O programa lê os dados via **entrada padrão (stdin)**:

```
M N
Q[0][0] Q[0][1] ... Q[0][N-1]
...                              ← M linhas para Q
K[0][0] K[0][1] ... K[0][N-1]
...                              ← M linhas para K
V[0][0] V[0][1] ... V[0][N-1]
...                              ← M linhas para V
```

| Parâmetro | Descrição                              | Restrição          |
|-----------|----------------------------------------|--------------------|
| `M`       | Número de linhas das matrizes Q, K, V  | 2 ≤ M ≤ 4096       |
| `N`       | Número de colunas (= dk)               | 2 ≤ N ≤ 4096       |
| Valores   | Elementos das matrizes em FP32         | 0.00 ≤ X ≤ 1.00    |

### Exemplo de uso com arquivo

```bash
./attention < entrada.txt
```

### Exemplo de entrada (`entrada.txt`)

```
2 3
0.1 0.2 0.3
0.4 0.5 0.6
0.7 0.8 0.9
0.1 0.2 0.3
0.3 0.5 0.2
0.6 0.1 0.8
```

### Saída

A matriz `Attention(Q, K, V)` com 4 casas decimais, sem espaço no final de cada linha:

```
0.3819 0.3330 0.5997
0.3897 0.3365 0.6039
```

Tolerância aceita: variação de até **±0.03** por elemento em relação à referência.

---

## Detalhes de implementação

### Parâmetros de configuração (`#define`)

| Constante | Valor | Descrição |
|-----------|-------|-----------|
| `VALX`    | 8     | Número de linhas processadas por bloco (Br) |
| `VALY`    | 256   | Largura do tile na dimensão N (Bc) |
| `MAXL`    | 4097  | Tamanho máximo de linha + 1 (guard) |
| `PADDING` | 1     | Padding para evitar bank conflicts na shmem |
| `MASK`    | 0xffffffff | Máscara para warp shuffle (`__shfl_down_sync`) |

### Kernel `meuAttention`

Implementa o algoritmo de atenção com tiling em duas fases, percorrendo a dimensão N em blocos de tamanho `VALY`:

**Fase 1 — Cálculo de QK^T e máximo de linha (S, row_mx):**
- Carrega tiles de `Q` e `K` na memória compartilhada.
- Para N divisível por VALY: usa vetorização com `float4` para maior largura de banda.
- Para o tile restante (borda): fallback escalar com verificação de bounds.
- Acumula o produto escalar usando `fmaf` e reduz com `__shfl_down_sync` (warp shuffle) para somar parciais entre threads do mesmo warp.
- Mantém o máximo de linha (`row_mx`) para estabilidade numérica do softmax.

**Fase 2 — Softmax online e acumulação de saída (O):**
- Aplica softmax numericamente estável: `exp(S[y] - row_mx)`, normalizado por `row_l_new`.
- Usa a recorrência do FlashAttention para atualizar `l` (soma de exponenciais) e `m` (máximo) de forma incremental a cada tile, sem materializar a matriz de atenção completa.
- Multiplica S (softmax parcial) por tiles de `V` e acumula em `O` com correção pelo fator `exp(m_prev - m_new)`.

### Kernel `initLM`

Inicializa os vetores auxiliares antes do loop de tiles:
- `l[i] = 0.0f` — acumulador da soma do softmax
- `m[i] = -1e9f` — máximo atual (inicializado com -infinito)

### Alinhamento de memória (N4)

As matrizes são alocadas com largura `N4 = ceil(N/4)*4` para garantir acesso alinhado a `float4`, melhorando a coalescência na memória global.

### Saída com buffer manual

A impressão usa um buffer de 16 MB com `fwrite` em vez de `printf` por elemento, evitando overhead de I/O para matrizes grandes.

### Funções CPU auxiliares (não usadas na saída final)

| Função        | Descrição |
|---------------|-----------|
| `MMA`         | Multiplicação de matrizes CPU (referência) |
| `MMAt`        | Multiplicação com transposição de B (CPU) |
| `softmax`     | Softmax estável por linha (CPU) |
| `checkMatrix` | Valida resultado GPU vs CPU com tolerância de 0.03 |
| `initMatrix`  | Inicializa matriz com valores aleatórios em [0,1] |

---

## Limitações conhecidas

- Apenas matrizes com **M = N** são suportadas corretamente pelo kernel (matrizes quadradas).
- `VALX` e `VALY` são fixos em tempo de compilação; ajustá-los pode ser necessário para GPUs com menor memória compartilhada.
- Não há verificação de erros CUDA nas chamadas de alocação e cópia (recomendado adicionar `CUDA_CHECK` para uso em produção; a macro já está definida no código).