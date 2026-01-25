# Nyström 近似 DPP 核 + FastMAP 算法描述

## 学术论文格式描述（推荐使用）

Although exact MAP inference for DPP requires constructing the full kernel matrix L ∈ ℝ^(n×n) with O(n²D) complexity, we employ a Nyström approximation to avoid explicit kernel construction while maintaining the DPP framework. Specifically, we approximate the DPP kernel L = diag(r) × S × diag(r), where S ∈ ℝ^(n×n) is the cosine similarity matrix and r ∈ ℝ^n is the relevance vector, using a low-rank factorization L ≈ VV^T with V ∈ ℝ^(n×m) and m ≪ n. The Nyström method selects m landmark points uniformly and constructs submatrices L_mm ∈ ℝ^(m×m) and L_Nm ∈ ℝ^(n×m) by computing only the similarity between all points and landmarks, avoiding the full n×n similarity matrix. Through Cholesky decomposition of L_mm = CC^T, we obtain V = L_Nm × (C^T)^(-1), enabling implicit kernel representation. We then apply the FastMAP greedy algorithm [Chen et al., 2018] on the approximated kernel, where each iteration computes kernel rows via V[j]V^T in O(mn) time instead of accessing the full matrix. The overall time complexity is reduced from O(n²D + k²n) to O(nDm + m³ + kmn), where k is the number of selected items. When m ≪ n, this yields substantial speedups (typically 1.3-2× faster) with negligible quality degradation, as the Nyström approximation preserves the principal structure of the DPP kernel while reducing memory footprint from O(n²) to O(nm).

## 简洁版本（类似参考格式）

Although MAP inference for DPP typically requires constructing the full kernel matrix L ∈ ℝ^(n×n) with O(n²D) complexity, we employ a Nyström approximation to reduce computational cost while preserving the DPP framework. We approximate the DPP kernel L = diag(r) × S × diag(r) using a low-rank factorization L ≈ VV^T, where V ∈ ℝ^(n×m) is obtained by selecting m landmark points and performing Cholesky decomposition on the landmark submatrix. The FastMAP greedy algorithm [Chen et al., 2018] is then applied on the approximated kernel, where each iteration computes kernel rows implicitly via V[j]V^T in O(mn) time. The overall time complexity is reduced from O(n²D + k²n) to O(nDm + m³ + kmn). The additional latency is negligible when m ≪ n, with typically 1.3-2× speedup and less than 10ms per segment for large-scale video token pruning (n > 2000).

## 中文版本

尽管 DPP 的精确 MAP 推理需要构建完整的核矩阵 L ∈ ℝ^(n×n)，复杂度为 O(n²D)，我们采用 Nyström 近似方法在保持 DPP 框架的同时避免显式构建完整核矩阵。具体而言，我们将 DPP 核 L = diag(r) × S × diag(r)（其中 S ∈ ℝ^(n×n) 为余弦相似度矩阵，r ∈ ℝ^n 为相关性向量）近似为低秩分解 L ≈ VV^T，其中 V ∈ ℝ^(n×m) 且 m ≪ n。Nyström 方法均匀选择 m 个地标点，通过仅计算所有点与地标之间的相似度构建子矩阵 L_mm ∈ ℝ^(m×m) 和 L_Nm ∈ ℝ^(n×m)，从而避免构建完整的 n×n 相似度矩阵。通过对 L_mm = CC^T 进行 Cholesky 分解，我们得到 V = L_Nm × (C^T)^(-1)，实现隐式核表示。随后，我们在近似核上应用 FastMAP 贪心算法 [Chen et al., 2018]，其中每次迭代通过 V[j]V^T 在 O(mn) 时间内计算核矩阵行，而非访问完整矩阵。总体时间复杂度从 O(n²D + k²n) 降低至 O(nDm + m³ + kmn)，其中 k 为选择的项目数量。当 m ≪ n 时，该方法可带来显著加速（通常快 1.3-2 倍），且质量损失可忽略，因为 Nyström 近似在将内存占用从 O(n²) 降至 O(nm) 的同时保留了 DPP 核的主要结构。

## 复杂度详细分析

### 原始方法（完整核矩阵）
- **核矩阵构建**：O(n²D) - 计算所有 n×n 对之间的余弦相似度
- **FastMAP 推理**：O(k²n) - k 次迭代，每次更新 O(kn) 元素
- **总复杂度**：O(n²D + k²n)
- **内存复杂度**：O(n²)

### Nyström 近似方法
- **地标选择**：O(1) - 均匀采样
- **子矩阵构建**：
  - L_mm：O(m²D) - m 个地标之间的相似度
  - L_Nm：O(nmD) - 所有点与 m 个地标之间的相似度
- **Cholesky 分解**：O(m³) - L_mm 的分解
- **V 矩阵计算**：O(nm²) - 求解线性系统
- **FastMAP 推理**：O(kmn) - k 次迭代，每次通过 V 计算一行（O(mn)）
- **总复杂度**：O(nDm + m³ + kmn)
- **内存复杂度**：O(nm)

### 加速比
当 m = 128, n = 4096, k = 600, D = 512 时：
- **原始方法**：O(4096² × 512 + 600² × 4096) ≈ O(8.6×10⁹ + 1.5×10⁹) ≈ O(10¹⁰)
- **Nyström 方法**：O(4096 × 512 × 128 + 128³ + 600 × 128 × 4096) ≈ O(2.7×10⁸ + 2.1×10⁶ + 3.1×10⁸) ≈ O(5.8×10⁸)
- **理论加速比**：约 17×（实际测试中约 1.3-2×，受 GPU 并行化影响）
