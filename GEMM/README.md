# Multiplicação de Matrizes com CUDA

Implementação de multiplicação de matrizes quadradas utilizando GPU via CUDA, com otimizações de memória compartilhada (*tiling* com múltiplos blocos K) e padding para evitar conflitos de banco de memória.

---

## Requisitos

- NVIDIA GPU com suporte a CUDA
- CUDA Toolkit instalado (recomendado: 11.x ou superior)
- Compilador `nvcc`
- Sistema operacional: Windows ou Linux

---

## Compilação

```bash
nvcc -O2 -o matriz matriz.cu
```

---

## Execução

O programa lê os dados via **entrada padrão (stdin)** no seguinte formato:

```
N
a[0][0] a[0][1] ... a[0][N-1]
...
a[N-1][0] ... a[N-1][N-1]
b[0][0] b[0][1] ... b[0][N-1]
...
b[N-1][0] ... b[N-1][N-1]
```

- **Linha 1:** inteiro `N` — dimensão das matrizes (N×N)
- **Próximas N linhas:** elementos da matriz **A** (floats separados por espaço)
- **Próximas N linhas:** elementos da matriz **B** (floats separados por espaço)

### Exemplo de uso com arquivo

```bash
./matriz < entrada.txt
```

### Exemplo de entrada (`entrada.txt`)

```
3
1.0 2.0 3.0
4.0 5.0 6.0
7.0 8.0 9.0
9.0 8.0 7.0
6.0 5.0 4.0
3.0 2.0 1.0
```

### Saída esperada

```
30.00 24.00 18.00
84.00 69.00 54.00
138.00 114.00 90.00
```

---

## Detalhes de implementação

### Parâmetros de configuração (`#define`)

| Constante    | Valor | Descrição                                         |
|--------------|-------|---------------------------------------------------|
| `TILE_SIZE`  | 16    | Dimensão do tile 2D (threads por bloco em x e y) |
| `TILE_SIZEK` | 8     | Número de sub-tiles ao longo da dimensão K        |
| `PADDING`    | 1     | Padding na memória compartilhada para evitar bank conflicts |

### Kernel `matrixMul`

- Utiliza **memória compartilhada** para carregar tiles das matrizes A e B, reduzindo acessos à memória global.
- O loop externo percorre tiles ao longo da dimensão K (colunas de A / linhas de B).
- O loop interno (`TILE_SIZEK`) acumula `TILE_SIZEK` sub-tiles por iteração, aumentando o reuso de dados em cache.
- Usa `fmaf` (fused multiply-add) para melhor precisão e desempenho.
- Sincronização com `__syncthreads()` garante consistência entre threads do mesmo bloco.

### Alocação de memória na GPU

Utiliza `cudaMallocPitch` para alocar memória 2D alinhada, o que pode melhorar a coalescência de acessos à memória global em alguns casos.

---

## Limitações conhecidas

- Apenas matrizes **quadradas** (N×N) são suportadas.
- O tamanho máximo de N é limitado pela memória da GPU disponível. (GTX 1650 4GB VRAM)
