# ML Vectorization & Linear Algebra Benchmark API

This repository empirically demonstrates the performance impact of **Linear Algebra vectorization** when training Machine Learning models (Logistic Regression with Gradient Descent and Binary Cross-Entropy loss).

We compare two distinct implementation approaches to solve the exact same mathematical problem across scaled datasets:

1. **Iterative Approach (Pure Python Loops):** Traditional element-wise processing using explicit loops.

2. **Vectorized Approach (NumPy + Linear Algebra):** Optimized matrix operations leveraging hardware-level capabilities.

## ⚡ Empirical Benchmark Results

Below are real execution metrics captured from live endpoint tests processing **20,000 samples**:

| Implementation | Samples | Execution Time | Processing Speed (Samples/sec) | Speedup Factor | Time Reduction (%) |
| ----- | ----- | ----- | ----- | ----- | ----- |
| **Unoptimized (Loops)** | 20,000 | **0.4309s** | \~46,414 samples/s | **1x (Baseline)** | **0% (Baseline)** |
| **Vectorized (NumPy)** | 20,000 | **0.0031s** | \~6,451,612 samples/s | **\~139x Faster** 🚀 | **99.28% Less Time** ⚡ |

### 📉 Time Reduction Breakdown (%)

To calculate the exact percentage of processing time saved by switching to Linear Algebra vectorization:

$$
\text{Time Saved (\%)} = \left( 1 - \frac{\text{Time}_{\text{Vectorized}}}{\text{Time}_{\text{Unoptimized}}} \right) \times 100
$$

$$
\text{Time Saved (\%)} = \left( 1 - \frac{0.0031\,\text{s}}{0.4309\,\text{s}} \right) \times 100 = (1 - 0.007194) \times 100 \approx \mathbf{99.28\%}
$$

* **Result:** The vectorized approach executed the optimization task using **99.28% less time** than the traditional loop implementation.

$$
\text{Speedup Multiple} = \frac{\text{Time}_{\text{Unoptimized}}}{\text{Time}_{\text{Vectorized}}} = \frac{0.4309\,\text{s}}{0.0031\,\text{s}} \approx \mathbf{139\times}
$$

## 📐 Mathematical Foundation

### 1. Prediction Model & Sigmoid Activation Function

Given an input matrix $X$ of shape $m \times n$ (where $m$ is the number of samples and $n$ is the number of features), a weight vector $W$ of shape $n \times 1$, and a bias scalar $b$, the linear combination and the predicted probability $\hat{Y}$ are defined as:

$$
Z = X \cdot W + b
$$

$$
\hat{Y} = \sigma(Z) = \frac{1}{1 + e^{-Z}}
$$

### 2. Loss Function: Binary Cross-Entropy

To measure the discrepancy between predictions $\hat{Y}$ and actual ground truth labels $Y \in \{0, 1\}$, we apply the Binary Cross-Entropy loss function:

$$
J(W, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]
$$

### 3. Gradient Descent & Vectorized Matrix Calculus

To minimize $J(W, b)$ and optimize parameters, we compute partial derivatives with respect to $W$ and $b$.

Applying the calculus chain rule yields a clean error term:

$$
E = \hat{Y} - Y
$$

#### Iterative Computation (Loops):

Computing gradients for each feature $j$ manually across all samples requires iterating element by element:

$$
\frac{\partial J}{\partial W_j} = \frac{1}{m} \sum_{i=1}^{m} ( \hat{y}^{(i)} - y^{(i)} ) x_j^{(i)}
$$

#### Vectorized Computation (Linear Algebra):

Instead of sample-by-sample summation, we perform matrix multiplication using the transpose $X^T$ (shape $n \times m$):

$$
\nabla_W J = \frac{1}{m} X^T \cdot E
$$

$$
\nabla_b J = \frac{1}{m} \sum E
$$

Parameter updates are performed simultaneously in block memory operations:

$$
W := W - \alpha \nabla_W J
$$

$$
b := b - \alpha \nabla_b J
$$

## 🚀 Why Vectorization Delivers Massive Speedups

1. **Elimination of Python Interpreter Overhead:** Python `for` loops incur significant dynamic type-checking and memory allocation overhead at every iteration.

2. **Low-Level Native Execution:** NumPy delegates matrix computations directly to C and Fortran optimized linear algebra libraries (such as BLAS and LAPACK).

3. **Hardware-Level Parallelism (SIMD):** Modern CPUs use **SIMD** (*Single Instruction, Multiple Data*) architecture, executing identical mathematical operations across large data vectors in a single clock cycle.

## 🛠️ Testing the Live Endpoints

You can benchmark the execution speeds directly via cURL commands:

### 1. Unoptimized Endpoint (Iterative Loops)

* **URL:** `http://162.35.24.220:8001/treinar-lento`

* **cURL Example:**

```bash
curl -X 'POST' \
  'http://162.35.24.220:8001/treinar-lento?qtd_amostras=20000' \
  -H 'accept: application/json' \
  -d ''
```

* **Sample Response:**

```json
{
  "metodo": "Sem otimização",
  "amostras_processadas": 20000,
  "tempo_segundos": 0.4309
}
```

### 2. Optimized Endpoint (Linear Algebra & Vectorization)

* **URL:** `http://162.35.24.220:8002/treinar-rapido`

* **cURL Example:**

```bash
curl -X 'POST' \
  'http://162.35.24.220:8002/treinar-rapido?qtd_amostras=20000' \
  -H 'accept: application/json' \
  -d ''
```

## 📊 Repository Structure

```text
├── api_sem_otimizacao.py    # FastAPI endpoint with iterative loop training
├── api_otimizada.py         # FastAPI endpoint with vectorized matrix training
└── README.md                # Documentation and mathematical background
```
