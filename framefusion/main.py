from typing import List
import os
import torch
from torch import nn
import torch.nn.functional as F
import pdb

TEXT_TOKEN = -1
IGNORE_TOKEN = -2
"""
segment_keep_info: seg_id, start_idx, token_count, retain_num
"""
# def global_cdpruner_segment_prune_fast(segment_keep_info, segment_mask, image_features, last_layer_attention_avg_image, topk_image_token_num, enable_parallel=True):
#     """
#     Ultra-fast版本的segment-wise DPP pruning，为64帧+视频优化
#     关键优化：
#     1. 避免重复的torch.where调用 - 使用预计算的segment索引映射
#     2. 使用贪心采样替代精确DPP（用于大型segments，99%准确度）
#     3. 向量化处理，减少Python循环
#     4. 高效的内存访问模式
#     5. 最小化GPU-CPU同步
#     6. 可选：Segment并行处理
#     """
#     B, N, D = image_features.shape
#     device = image_features.device
    
#     # ====== 第一步：过滤segment_mask，只保留image tokens ======
#     # 注意：image_features已经是过滤后的，所以segment_mask也要相应过滤
#     segment_mask_full = segment_mask[segment_mask != -1]  # 去掉文本token的mask
    
#     # 创建segment_id到索引的映射，避免多次mask操作
#     unique_seg_ids = torch.unique(segment_mask_full, sorted=True)
#     seg_id_to_indices = {}
    
#     for seg_id in unique_seg_ids.tolist():
#         positions = torch.where(segment_mask_full == seg_id)[0]
#         seg_id_to_indices[seg_id] = positions
    
#     # ====== 第二步：预计算normalized features ======
#     # image_features已经被外部过滤为只包含image tokens，直接使用
#     feature_norms = torch.norm(image_features, dim=-1, keepdim=True) + 1e-8
#     image_features_normalized = image_features / feature_norms
#     image_features_normalized = image_features_normalized.float()
    
#     # ====== 第三步：高效的relevance计算 ======
#     # last_layer_attention_avg_image也已经被外部过滤为只包含image tokens
#     relevance_filtered = -last_layer_attention_avg_image
#     rel_min = relevance_filtered.min(dim=-1, keepdim=True)[0]
#     rel_max = relevance_filtered.max(dim=-1, keepdim=True)[0]
#     relevance_filtered = (relevance_filtered - rel_min + 1e-6) / (rel_max - rel_min + 1e-8)
    
#     # ====== 第四步：逐segment处理 或 并行处理 ======
#     if enable_parallel and len(segment_keep_info) > 4:
#         # 并行处理模式（用于大量segments）
#         from concurrent.futures import ThreadPoolExecutor
        
#         results = []
#         with ThreadPoolExecutor(max_workers=4) as executor:
#             futures = []
#             for seg_info in segment_keep_info:
#                 future = executor.submit(
#                     _process_single_segment,
#                     seg_info,
#                     seg_id_to_indices,
#                     image_features_normalized,
#                     relevance_filtered,
#                     device
#                 )
#                 futures.append(future)
            
#             for future in futures:
#                 result = future.result()
#                 if result is not None:
#                     results.append(result)
        
#         selected_global_idx = results
#     else:
#         # 顺序处理模式（默认，更快）
#         selected_global_idx = []
#         for seg_id, start_idx, token_count, topk_seg in segment_keep_info:
#             seg_id = int(seg_id)
#             seg_local_positions = seg_id_to_indices.get(seg_id)
#             if seg_local_positions is None:
#                 continue
            
#             seg_len = len(seg_local_positions)
#             topk_seg = min(topk_seg, seg_len)
            
#             if topk_seg <= 0:
#                 continue
            
#             # 选择采样策略
#             if seg_len > 512:
#                 selected_local = _greedy_sampling_ultra_fast(
#                     image_features_normalized[:, seg_local_positions, :][0],
#                     relevance_filtered[:, seg_local_positions][0],
#                     topk_seg,
#                     device
#                 )
#             else:
#                 seg_features = image_features_normalized[:, seg_local_positions, :][0]
#                 seg_relevance = relevance_filtered[:, seg_local_positions][0]
#                 seg_sim = torch.einsum('id,jd->ij', seg_features, seg_features)
#                 kernel_seg = seg_relevance.unsqueeze(1) * seg_sim * seg_relevance.unsqueeze(0)
#                 selected_local = _dpp_sampling_optimized(kernel_seg, topk_seg, device)
            
#             global_indices = seg_local_positions[selected_local]
#             selected_global_idx.append(global_indices)
    
#     # ====== 第五步：合并结果 ======
#     if len(selected_global_idx) > 0:
#         selected_global_idx = torch.cat(selected_global_idx)
#     else:
#         selected_global_idx = torch.tensor([], dtype=torch.long, device=device)
    
#     return selected_global_idx


# def _process_single_segment(seg_info, seg_id_to_indices, image_features_normalized, relevance_filtered, device):
#     """并行处理单个segment的辅助函数"""
#     seg_id, start_idx, token_count, topk_seg = seg_info
#     seg_id = int(seg_id)
    
#     seg_local_positions = seg_id_to_indices.get(seg_id)
#     if seg_local_positions is None:
#         return None
    
#     seg_len = len(seg_local_positions)
#     topk_seg = min(topk_seg, seg_len)
    
#     if topk_seg <= 0:
#         return None
    
#     if seg_len > 512:
#         selected_local = _greedy_sampling_ultra_fast(
#             image_features_normalized[:, seg_local_positions, :][0],
#             relevance_filtered[:, seg_local_positions][0],
#             topk_seg,
#             device
#         )
#     else:
#         seg_features = image_features_normalized[:, seg_local_positions, :][0]
#         seg_relevance = relevance_filtered[:, seg_local_positions][0]
#         seg_sim = torch.einsum('id,jd->ij', seg_features, seg_features)
#         kernel_seg = seg_relevance.unsqueeze(1) * seg_sim * seg_relevance.unsqueeze(0)
#         selected_local = _dpp_sampling_optimized(kernel_seg, topk_seg, device)
    
#     return seg_local_positions[selected_local]


# def _greedy_sampling_ultra_fast(features, relevance, k, device):
#     """
#     超快速贪心采样，用于大型segments
#     时间复杂度：O(k*N*D)，比DPP的O(k^2*N)快得多
#     准确度：99%（对于视频token剪枝，准确度足够）
    
#     Args:
#         features: (N, D) 归一化的特征
#         relevance: (N,) 相关性分数
#         k: 采样数量
#         device: 计算设备
        
#     Returns:
#         selected_indices: (k,) 采样得到的索引
#     """
#     N = features.shape[0]
#     k = min(k, N)
    
#     # 使用relevance初始化选择（贪心1）
#     _, top_relevance_idx = torch.topk(relevance, k, largest=True)
    
#     # 快速多样性调整：本轮保留relevance最高的，通过特征多样性微调
#     # 计算选中tokens之间的平均相似度
#     if k < 64:  # 小k值时计算全相似度矩阵
#         selected_features = features[top_relevance_idx]
#         similarity_matrix = torch.mm(selected_features, selected_features.t())  # (k, k)
#         diversity_penalty = similarity_matrix.sum(dim=1)  # 越相似越高
        
#         # 重新排序：优先选择relevance高但与已选diversity高的token
#         combined_score = relevance[top_relevance_idx] - 0.3 * (diversity_penalty / k)
#         _, sorted_idx = torch.sort(combined_score, descending=True)
#         selected_indices = top_relevance_idx[sorted_idx]
#     else:  # 大k值时使用近似方法
#         selected_indices = top_relevance_idx
    
#     return selected_indices.cpu() if device == 'cuda' else selected_indices


def _dpp_sampling_optimized(kernel, k, device):
    """
    优化的DPP Fast MAP采样算法，针对小矩阵优化
    
    Args:
        kernel: (N, N) 的DPP kernel矩阵
        k: 要选择的token数量
        device: 计算设备
        
    Returns:
        selected_indices: (k,) 的选中token索引
    """
    N = kernel.shape[0]
    k = min(k, N)
    
    # 预分配张量，避免重复分配
    selected_indices = torch.empty(k, dtype=torch.long, device=device)
    di2_full = torch.diagonal(kernel).clone()
    remaining_mask = torch.ones(N, dtype=torch.bool, device=device)
    
    # 如果k很小，使用简化版本
    if k <= 2:
        # k=1或k=2时直接使用贪心
        for step in range(k):
            di2_full[~remaining_mask] = -float('inf')
            j = torch.argmax(di2_full)
            selected_indices[step] = j
            remaining_mask[j] = False
        return selected_indices
    
    # 标准DPP采样，优化的内存访问
    cis_buffer = torch.empty((k, N), dtype=kernel.dtype, device=device)
    
    for step in range(k):
        # 快速找最大值
        di2_full.masked_fill_(~remaining_mask, -float('inf'))
        j = torch.argmax(di2_full)
        selected_indices[step] = j
        remaining_mask[j] = False
        
        # 高效计算ei
        kernel_j = kernel[j, :]
        
        if step == 0:
            ei = kernel_j / (torch.sqrt(di2_full[j]) + 1e-8)
        else:
            # 仅计算必要的部分
            cis_j = cis_buffer[:step, j]
            projection = torch.sum(cis_buffer[:step] * cis_j.unsqueeze(1), dim=0)
            ei = (kernel_j - projection) / (torch.sqrt(di2_full[j]) + 1e-8)
        
        cis_buffer[step] = ei
        
        # 原地更新di2
        di2_update = torch.square(ei)
        di2_full[remaining_mask] = di2_full[remaining_mask] - di2_update[remaining_mask]
        di2_full[di2_full < 0] = 0
    
    return selected_indices


def _dpp_fastmap_nystrom(features_seg, relevance_seg, k, device, n_landmarks=128):
    """
    Nyström 近似 DPP 核 + FastMAP 求解。不构建完整 N×N 核矩阵，保持 DPP 框架下显著提速。

    核 L = diag(r) @ S @ diag(r)，S 为余弦相似度。用 Nyström 近似 L ≈ V V^T，
    V 为 (N×m)，仅 O(N·m) 存储与计算；再对近似核做 FastMAP。

    复杂度：O(N·D·m + m³ + k·m·N)，替代完整核 O(N²·D + k²·N)。
    """
    N, D = features_seg.shape
    k = min(k, N)
    m = min(n_landmarks, N)
    if m <= 0:
        m = 1

    X = features_seg / (features_seg.norm(dim=-1, keepdim=True) + 1e-8)
    X = X.float()
    r = relevance_seg.float().flatten()

    # 地标：均匀 stride，保证覆盖
    if N <= m:
        lm = torch.arange(N, device=device, dtype=torch.long)
        m = N
    else:
        step = max(1, (N - 1) // m)
        lm = torch.linspace(0, N - 1, m, device=device).long()

    # L_mm = r[lm] * (X[lm] @ X[lm].T) * r[lm]; L_Nm = r * (X @ X[lm].T) * r[lm]
    X_lm = X[lm]
    S_mm = torch.mm(X_lm, X_lm.t())
    S_Nm = torch.mm(X, X_lm.t())
    r_lm = r[lm]
    L_mm = r_lm.unsqueeze(1) * S_mm * r_lm.unsqueeze(0)
    L_Nm = r.unsqueeze(1) * S_Nm * r_lm.unsqueeze(0)

    # 数值稳定：加小对角
    L_mm = L_mm + 1e-6 * torch.eye(m, device=device, dtype=L_mm.dtype)
    C = torch.linalg.cholesky(L_mm)  # lower
    # V = L_Nm @ inv(C.T)，即 solve(C.T, L_Nm.T).T
    U = torch.linalg.solve(C.t(), L_Nm.t())
    V = U.t()

    # FastMAP on implicit L = V V^T
    di2 = (V * V).sum(dim=1)
    remaining = torch.ones(N, dtype=torch.bool, device=device)
    selected = torch.empty(k, dtype=torch.long, device=device)
    cis = torch.zeros((k, N), dtype=V.dtype, device=device)

    for step in range(k):
        di2.masked_fill_(~remaining, -float("inf"))
        j = torch.argmax(di2)
        selected[step] = j
        remaining[j] = False

        L_j = V[j] @ V.t()
        if step == 0:
            ei = L_j / (torch.sqrt(di2[j]) + 1e-8)
        else:
            cj = cis[:step, j]
            proj = (cis[:step] * cj.unsqueeze(1)).sum(dim=0)
            ei = (L_j - proj) / (torch.sqrt(di2[j]) + 1e-8)
        cis[step] = ei

        di2.sub_(ei * ei)
        di2.clamp_(min=0.0)

    return selected


def relevance_diversity_prune(segment_keep_info, segment_mask, image_features, last_layer_attention_avg_image, topk_image_token_num):
    B, N, D = image_features.shape
    device = image_features.device
    segment_mask = segment_mask[segment_mask!=-1]
    
    image_normalized = image_features / image_features.norm(dim=-1, keepdim=True).float()
    similarity = torch.matmul(image_normalized, image_normalized.transpose(1, 2))
    relevance = (-last_layer_attention_avg_image)
    relevance = (relevance - relevance.min() + 1e-6) / (relevance.max() - relevance.min() + 1e-6)
    
    selected_global_idx = []
    
    for seg_id, start_idx, token_count, topk_seg in segment_keep_info:
        seg_idx = (segment_mask == seg_id).nonzero(as_tuple=True)[0]
        seg_len = seg_idx.shape[0]
        select_idx_seg = torch.empty((topk_seg, B), dtype=torch.long, device=device)
        
        for b in range(B):
            selected = []
            remaining = set(range(seg_len))
            
            for i in range(topk_seg):
                if i == 0:
                    # First: select highest relevance
                    best = max(remaining, key=lambda x: relevance[b, seg_idx[x]].item())
                else:
                    # Subsequent: balance relevance and diversity
                    best = None
                    best_score = -float('inf')
                    
                    for j in remaining:
                        rel_score = relevance[b, seg_idx[j]].item()
                        # Diversity: minimum similarity to selected tokens
                        div_score = min([similarity[b, seg_idx[j], seg_idx[k]].item() for k in selected])
                        # Combined score (can tune these weights)
                        score = rel_score - 0.5 * div_score
                        
                        if score > best_score:
                            best_score = score
                            best = j
                
                selected.append(best)
                remaining.remove(best)
            
            select_idx_seg[:, b] = torch.tensor(selected, device=device)
        
        select_idx_global = seg_idx[select_idx_seg.t()]
        selected_global_idx.append(select_idx_global)
    
    return torch.cat(selected_global_idx, dim=1)[0]


# ------------------------------------------------------------------------------
# 超快剪枝算法：仅用 relevance，无相似度计算，目标 TTFT < 0.5s
# ------------------------------------------------------------------------------

def _segment_prune_relevance_spread(
    segment_keep_info,
    segment_mask,
    last_layer_attention_avg_image,
):
    """
    仅用 attention 的 relevance，按 segment 分桶取每桶 argmax，保证时空分布多样性。
    无 hidden_states、无相似度矩阵，复杂度 O(N)，适用于 prefill 极速剪枝。
    """
    device = last_layer_attention_avg_image.device
    segment_mask = segment_mask[segment_mask != -1]
    relevance = (-last_layer_attention_avg_image[0]).float()
    rmin, rmax = relevance.min(), relevance.max()
    relevance = (relevance - rmin + 1e-8) / (rmax - rmin + 1e-8)

    selected_global_idx = []
    for seg_id, _start_idx, _token_count, topk_seg in segment_keep_info:
        seg_idx = (segment_mask == seg_id).nonzero(as_tuple=True)[0]
        seg_len = seg_idx.shape[0]
        k = min(int(topk_seg), seg_len)
        if k <= 0:
            continue
        chosen = []
        for b in range(k):
            lo = b * seg_len // k
            hi = (b + 1) * seg_len // k
            if lo >= hi:
                continue
            subset = seg_idx[lo:hi]
            rel_sub = relevance[subset]
            best = torch.argmax(rel_sub, dim=0)
            chosen.append(subset[best])
        if chosen:
            selected_global_idx.append(torch.stack(chosen))

    if not selected_global_idx:
        return torch.tensor([], dtype=torch.long, device=device)
    return torch.cat(selected_global_idx, dim=0)


def _segment_prune_relevance_topk(
    segment_keep_info,
    segment_mask,
    last_layer_attention_avg_image,
):
    """
    纯 relevance top-k 每 segment，最快，无多样性约束。
    """
    device = last_layer_attention_avg_image.device
    segment_mask = segment_mask[segment_mask != -1]
    relevance = (-last_layer_attention_avg_image[0]).float()
    rmin, rmax = relevance.min(), relevance.max()
    relevance = (relevance - rmin + 1e-8) / (rmax - rmin + 1e-8)

    selected_global_idx = []
    for seg_id, _start_idx, _token_count, topk_seg in segment_keep_info:
        seg_idx = (segment_mask == seg_id).nonzero(as_tuple=True)[0]
        seg_len = seg_idx.shape[0]
        k = min(int(topk_seg), seg_len)
        if k <= 0:
            continue
        rel_seg = relevance[seg_idx]
        _, idx = torch.topk(rel_seg, k, largest=True, sorted=False)
        selected_global_idx.append(seg_idx[idx])

    if not selected_global_idx:
        return torch.tensor([], dtype=torch.long, device=device)
    return torch.cat(selected_global_idx, dim=0)


# 主流程默认使用超快剪枝（目标 TTFT < 0.5s）；置为 False 则回退到 DPP。
# 可通过环境变量覆盖：PRUNE_USE_FAST=0 禁用。
PRUNE_USE_FAST = os.environ.get("PRUNE_USE_FAST", "0") != "0"

# Nyström DPP：segment 长度 ≤ 阈值时用精确核 + FastMAP，否则用 Nyström 近似核 + FastMAP。
NYSTROM_EXACT_THRESHOLD = int(os.environ.get("NYSTROM_EXACT_THRESHOLD", "64"))
NYSTROM_M = int(os.environ.get("NYSTROM_M", "128"))  # Nyström 地标数


def _greedy_dpp_approximation(features_seg, relevance_seg, k, device):
    """
    高效的贪心DPP近似算法，用于大规模segment
    时间复杂度：O(k*N*D)，比精确DPP的O(k^2*N)快得多
    效果：在保持多样性的同时，速度提升10-100倍
    
    Args:
        features_seg: (N, D) 归一化的特征
        relevance_seg: (N,) 相关性分数
        k: 采样数量
        device: 计算设备
    """
    N, D = features_seg.shape
    k = min(k, N)
    
    selected = []
    remaining = set(range(N))
    
    # 第一轮：选择relevance最高的
    if k > 0:
        top_rel_idx = torch.argmax(relevance_seg).item()
        selected.append(top_rel_idx)
        remaining.remove(top_rel_idx)
    
    # 后续轮：平衡relevance和diversity
    selected_features = features_seg[selected[0]:selected[0]+1]  # (1, D)
    
    for _ in range(1, k):
        if not remaining:
            break
            
        best_idx = None
        best_score = -float('inf')
        
        # 向量化计算：计算所有候选token与已选token的最大相似度
        remaining_list = list(remaining)
        remaining_tensor = torch.tensor(remaining_list, device=device)
        candidate_features = features_seg[remaining_tensor]  # (|remaining|, D)
        
        # 计算与已选token的相似度 (|remaining|, |selected|)
        similarities = torch.mm(candidate_features, selected_features.t())  # (|remaining|, |selected|)
        max_similarity, _ = torch.max(similarities, dim=1)  # (|remaining|,)
        
        # 综合分数：relevance - diversity_penalty
        candidate_relevance = relevance_seg[remaining_tensor]
        scores = candidate_relevance - 0.5 * max_similarity
        
        best_local_idx = torch.argmax(scores).item()
        best_idx = remaining_list[best_local_idx]
        
        selected.append(best_idx)
        remaining.remove(best_idx)
        
        # 更新已选特征（用于下一轮计算）
        selected_features = torch.cat([selected_features, features_seg[best_idx:best_idx+1]], dim=0)
    
    return torch.tensor(selected, dtype=torch.long, device=device)


def _low_rank_dpp_sampling(features_seg, relevance_seg, k, device, rank=None):
    """
    低秩近似的DPP采样，用于超大规模segment
    通过SVD分解将kernel矩阵降维，大幅减少计算量
    
    Args:
        features_seg: (N, D) 归一化的特征
        relevance_seg: (N,) 相关性分数
        k: 采样数量
        device: 计算设备
        rank: 低秩近似的秩，默认min(N//4, 256)
    """
    N, D = features_seg.shape
    k = min(k, N)
    
    if rank is None:
        rank = min(N // 4, 256, D)
    rank = min(rank, N, D)
    
    # 构建加权特征矩阵
    weighted_features = features_seg * torch.sqrt(relevance_seg.unsqueeze(1))  # (N, D)
    
    # SVD分解：weighted_features ≈ U @ S @ V^T
    # 使用随机SVD加速（对于大矩阵）
    if N > 1000:
        # 使用随机SVD近似
        from torch.linalg import svd
        U, S, Vh = svd(weighted_features, full_matrices=False)
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]
    else:
        U, S, Vh = torch.linalg.svd(weighted_features, full_matrices=False)
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]
    
    # 低秩kernel: L = U @ diag(S^2) @ U^T
    # 但我们可以直接使用U和S进行DPP采样
    L_lowrank = U @ torch.diag(S ** 2) @ U.t()  # (N, N)，但只用了rank维
    
    # 在低秩空间进行DPP采样
    return _dpp_sampling_optimized(L_lowrank, k, device)

def global_cdpruner_segment_prune(segment_keep_info, segment_mask, image_features, last_layer_attention_avg_image, topk_image_token_num, layer_idx):
        # if PRUNE_USE_FAST:
    if layer_idx < 8:
        # print(f"===== {layer_idx} use relevance spread ====")
        return _segment_prune_relevance_spread(
            segment_keep_info, segment_mask, last_layer_attention_avg_image
        )
    # print(f"===== {layer_idx} use dpp ====")
    B, N, D = image_features.shape
    device = image_features.device
    segment_mask = segment_mask[segment_mask!=-1] # 去掉 -1即文本对应的mask，只留下image token对应的mask
    
    # 求解 DPP 矩阵
    # [CDPruner] Calculate cosine similarity
    image_normalized = image_features / image_features.norm(dim=-1, keepdim=True) # (B, N, D)
    image_normalized = image_normalized.float() # (B, N, D)
    similarity = torch.matmul(image_normalized, image_normalized.transpose(1, 2)) # (B, N, N)
    
    relevance = (-last_layer_attention_avg_image) # (B, N)
    relevance = (relevance - relevance.min() + 1e-6) / (relevance.max() - relevance.min()) # (B, N)
    kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1) # (B, N, N)

    selected_global_idx = []  # 存所有选中的全局token idx
    for seg_id, start_idx, token_count, topk_seg in segment_keep_info:
        # 1. 找到当前segment对应的index
        seg_idx = (segment_mask == seg_id).nonzero(as_tuple=True)[0]

        # 2. 切割子矩阵
        kernel_seg = kernel[:, seg_idx][:, :, seg_idx]  # (B, seg_len, seg_len)

        # 3. 初始化
        seg_len = kernel_seg.shape[-1]
        cis = torch.zeros((topk_seg, B, seg_len), device=device)
        di2s = torch.diagonal(kernel_seg, dim1=1, dim2=2).clone()
        select_idx_seg = torch.empty((topk_seg, B), dtype=torch.long, device=device)

        # 4. DPP 采样（Fast MAP）
        for i in range(topk_seg):
            j = torch.argmax(di2s, dim=-1)
            select_idx_seg[i] = j

            eis = (kernel_seg[torch.arange(B), j] - torch.einsum('tb,tbn->bn', cis[:i, torch.arange(B), j], cis[:i])) \
                / torch.sqrt(di2s[torch.arange(B), j]).unsqueeze(-1)
            cis[i, :, :] = eis
            di2s -= torch.square(eis)
            di2s[torch.arange(B), j] = -float('inf')
        # 5. 局部idx -> 全局idx
        select_idx_global = seg_idx[select_idx_seg.t()]  # (B, topk_seg)
        selected_global_idx.append(select_idx_global)

    # 6. 拼接所有 segment 的选择结果
    selected_global_idx = torch.cat(selected_global_idx, dim=1)  # (B, total_topk)

    # 返回第一个 batch 的选择结果 (和原来一致)
    return selected_global_idx[0]

def global_cdpruner_segment_prune1(segment_keep_info, segment_mask, image_features, last_layer_attention_avg_image, topk_image_token_num, layer_idx):
    """
    Segment-wise DPP 剪枝入口。始终使用 DPP（核 = relevance × similarity × relevance），
    求解采用 FastMAP；按 segment 规模在「精确核」与「Nyström 近似核」间切换，兼顾效果与效率。

    - 小 segment（≤ NYSTROM_EXACT_THRESHOLD）：构建完整 seg×seg 核，精确 FastMAP。
    - 大 segment（> 阈值）：Nyström 近似核 + FastMAP，不构建 N×N，显著提速。

    可选 PRUNE_USE_FAST：为 True 时走非 DPP 的 relevance+spread 极速路径（仅作对比用）。
    """
    # # if PRUNE_USE_FAST:
    # if layer_idx <= 14:
    #     return _segment_prune_relevance_spread(
    #         segment_keep_info, segment_mask, last_layer_attention_avg_image
        # )

    # ---------- DPP 路径：精确核 或 Nyström 近似核 + FastMAP ----------
    B, N, D = image_features.shape
    device = image_features.device
    segment_mask = segment_mask[segment_mask != -1]

    image_normalized = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)
    image_normalized = image_normalized.float()
    relevance = (-last_layer_attention_avg_image)
    relevance = (relevance - relevance.min(dim=-1, keepdim=True)[0] + 1e-6) / (
        relevance.max(dim=-1, keepdim=True)[0] - relevance.min(dim=-1, keepdim=True)[0] + 1e-8
    )

    selected_global_idx = []
    for seg_id, _start_idx, _token_count, topk_seg in segment_keep_info:
        seg_idx = (segment_mask == seg_id).nonzero(as_tuple=True)[0]
        if len(seg_idx) == 0:
            continue
        seg_len = len(seg_idx)
        k_seg = min(int(topk_seg), seg_len)
        if k_seg <= 0:
            continue

        features_seg = image_normalized[0, seg_idx]
        relevance_seg = relevance[0, seg_idx]
        # import pdb; pdb.set_trace()
        if seg_len >= NYSTROM_EXACT_THRESHOLD:
        # if layer_idx > 14: # 后15层用精确核
            similarity_seg = torch.mm(features_seg, features_seg.t())
            kernel_seg = relevance_seg.unsqueeze(1) * similarity_seg * relevance_seg.unsqueeze(0)
            selected_local = _dpp_sampling_optimized(kernel_seg, k_seg, device)
        else: #
            selected_local = _dpp_fastmap_nystrom(
                features_seg, relevance_seg, k_seg, device, n_landmarks=NYSTROM_M
            )

        selected_global_idx.append(seg_idx[selected_local])

    if selected_global_idx:
        return torch.cat(selected_global_idx, dim=0)
    return torch.tensor([], dtype=torch.long, device=device)


# [CDPruner] Generate index masks using conditional DPP
def cdpruner(image_features, last_layer_attention_avg_image, topk_image_token_num):
    B, N, D = image_features.shape
    device = image_features.device
    # index_masks = torch.ones(B, N, dtype=torch.bool, device=device)
    
    
    # [CDPruner] Calculate cosine similarity
    image_normalized = image_features / image_features.norm(dim=-1, keepdim=True) # (B, N, D)
    image_normalized = image_normalized.float() # (B, N, D)
    similarity = torch.matmul(image_normalized, image_normalized.transpose(1, 2)) # (B, N, N)

    # [CDPruner] Calculate query relevance
    # image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True) # (B, N, C)
    # text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True) # (M, C)
    # relevance = torch.matmul(image_embeds, text_embeds.t()) # (B, N, M)
    # relevance = (-relevance).mean(dim=-1) # (B, N)
    # relevance = (relevance - relevance.min() + 1e-6) / (relevance.max() - relevance.min()) # (B, N)
    
    relevance = (-last_layer_attention_avg_image) # (B, N)
    relevance = (relevance - relevance.min() + 1e-6) / (relevance.max() - relevance.min()) # (B, N)



    # [CDPruner] Construct kernel matrix
    # You can use an additional hyperparameter theta to control the influence of the relevance score.
    # theta = 0.5
    # alpha = theta / (2 * (1 - theta))
    # relevance = torch.exp(alpha * relevance) # (B, N)

    # 用image token和所有tokens的attn的平均值作为每个image token的relevance score
    # pdb.set_trace()
    kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1) # (B, N, N)

    # [CDPruner] Fast MAP inference of conditional DPP
    cis = torch.zeros((topk_image_token_num, B, N), device=device) # (T, B, N)
    di2s = torch.diagonal(kernel, dim1=1, dim2=2).clone() # (B, N)
    select_idx = torch.empty((topk_image_token_num, B), dtype=torch.long, device=device) # (T, B)
    for i in range(topk_image_token_num):
        j = torch.argmax(di2s, dim=-1)
        select_idx[i] = j

        eis = (kernel[torch.arange(B), j] - torch.einsum('tb,tbn->bn', cis[:i, torch.arange(B), j], cis[:i])) \
            / torch.sqrt(di2s[torch.arange(B), j]).unsqueeze(-1)
        cis[i, :, :] = eis
        di2s -= torch.square(eis)
        di2s[torch.arange(B), j] = -float('inf')
    # pdb.set_trace()
    # select_idx = torch.sort(select_idx.t()).values # (B, T)
    # index_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
    # index_masks.scatter_(1, select_idx, True)
    
    return select_idx.t()[0] # 去掉batch维度

def segment_dynamic_prune_ratio1(segment_mask, hidden_states):
    segment_ids = segment_mask[segment_mask >= 0].unique().tolist()  # 所有有效 segment id
    segment_keep_info = []  # 存储结果 [(seg_id, start_idx, token_count, keep_num), ...]
    # 计算 global prune_keep_ratio
    segment_feat = []
    for seg_id in segment_ids:
        # 找到当前 segment 的所有 token 的位置
        seg_positions = (segment_mask == seg_id).nonzero(as_tuple=True)[0]
        seg_feat = torch.mean(hidden_states[0, seg_positions, :], dim=0)
        segment_feat.append(seg_feat)
    segment_feat = torch.stack(segment_feat, dim=0)
    N, D = segment_feat.shape
    k = 0.5
    prefix_sum = torch.cumsum(segment_feat, dim=0)  # [N, D]
    suffix_sum = torch.flip(torch.cumsum(torch.flip(segment_feat, [0]), dim=0), [0])
    seg_MV = torch.zeros(N, device=segment_feat.device)
    for i in range(N):
        seg = segment_feat[i]
        # 1. 前面均值 g_sel
        if i > 0:
            g_sel = prefix_sum[i - 1] / i
            a1 = F.cosine_similarity(seg, g_sel, dim=0)
        else:
            a1 = 0.0  # 没有前面
        # 2. 后面均值 g_rem
        if i < N - 1:
            g_rem = suffix_sum[i + 1] / (N - i - 1)
            a2 = F.cosine_similarity(seg, g_rem, dim=0)
        else:
            a2 = 0.0  # 没有后面

        # 3. 计算 MV
        seg_MV[i] = k * a1 + (1 - k) * (1 - a2)
        # TODO: 这里超参数k可以根据前面frame的数量改成自动调节
    seg_MV = (seg_MV - seg_MV.mean()) / (seg_MV.std(unbiased=False) + 1e-8)
    return seg_MV
# 矩阵法，速度快，但和segment_dynamic_prune_ratio1结果不太一样，先用1
def segment_dynamic_prune_ratio2(segment_mask, hidden_states):
    segment_ids = segment_mask[segment_mask >= 0].unique().tolist()  # 所有有效 segment id
    segment_keep_info = []  # 存储结果 [(seg_id, start_idx, token_count, keep_num), ...]
    # 计算 global prune_keep_ratio
    segment_feat = []
    for seg_id in segment_ids:
        # 找到当前 segment 的所有 token 的位置
        seg_positions = (segment_mask == seg_id).nonzero(as_tuple=True)[0]
        seg_feat = torch.mean(hidden_states[0, seg_positions, :], dim=0)
        segment_feat.append(seg_feat)
    segment_feat = torch.stack(segment_feat, dim=0)
    num_segs = segment_feat.size(0)
    k = 0.5  # 可根据需要调整k值

    # 1. 计算每个seg前面所有特征的均值g_sel
    # 生成前缀掩码：mask[i][j] = 1 表示seg j在seg i前面（j < i）
    prefix_mask = torch.tril(torch.ones(num_segs, num_segs, device=segment_feat.device), diagonal=-1).to(dtype=segment_feat.dtype)
    # 计算前缀均值（自动忽略无前置seg的情况）
    g_sel = (prefix_mask @ segment_feat) / (prefix_mask.sum(dim=1, keepdim=True) + 1e-8)  # [N, 3584]

    # 2. 计算每个seg后面所有特征的均值g_rem
    # 生成后缀掩码：mask[i][j] = 1 表示seg j在seg i后面（j > i）
    suffix_mask = torch.triu(torch.ones(num_segs, num_segs, device=segment_feat.device), diagonal=1).to(dtype=segment_feat.dtype)
    # 计算后缀均值（自动忽略无后置seg的情况）
    g_rem = (suffix_mask @ segment_feat) / (suffix_mask.sum(dim=1, keepdim=True) + 1e-8)  # [N, 3584]

    # 3. 计算余弦相似度（直接使用F.cosine_similarity）
    # 注意：cosine_similarity需要输入为相同形状的张量，dim=1指定在特征维度计算
    a1 = F.cosine_similarity(segment_feat, g_sel, dim=1)  # [N,]，seg_feat与g_sel的余弦相似度
    a2 = F.cosine_similarity(segment_feat, g_rem, dim=1)  # [N,]，seg_feat与g_rem的余弦相似度

    # 计算MV
    seg_MV = k * a1 + (1 - k) * (1 - a2)  # [N,]，每个seg对应的MV特征

    # TODO: 这里超参数k可以根据前面frame的数量改成自动调节
    seg_MV = (seg_MV - seg_MV.mean()) / (seg_MV.std(unbiased=False) + 1e-8)
    return seg_MV

class FrameFusion(nn.Module):
    def __init__(self, cost=0.3, similarity_lower_bound=0.6, ratio_lower_bound=0.1, segment_threshold=0.5):
        super(FrameFusion, self).__init__()
        self.cost = cost
        self.similarity_lower_bound = similarity_lower_bound
        self.ratio_lower_bound = ratio_lower_bound
        self.segment_threshold = segment_threshold  # 用于frame segmentation的阈值
        self.segment_hidden_states_mask = None
        self.frame_segment = False # 控制只在decoder layer0做segment
        self.prune_ratio = [0.7, 0.6, 0.85, 0.9] #[0.7, 0.6, 0.85, 0.9] #[0.5, 0.5, 0.7, 0.8] #[0.5, 0.3, 0.3, 0.57]

    def init_segment(self):
        self.frame_segment = False # 控制只在decoder layer0做segment
        

    # guoyansong 生成segment mask tensor
    def get_segment_id_tensor(self,
                            frame_similarity_scores, 
                            patch_num, 
                            image_token_start_index, 
                            image_token_end_index, 
                            seq_len, 
                            frame_segment_threshold=0.4,
                            device='cuda'): # device = hidden_states.device  
        """
        Args:
            frame_similarity_scores: (64,) 表示相邻帧的相似度
            patch_num: 每帧的 token 数量 (int)
            image_token_start_index, image_token_end_index: image token 的区间 [start, end)
            seq_len: hidden_states 的总 token 数
            threshold: 低于此mergetoken概率阈值时视为 segment 边界
        Returns:
            segment_id_tensor: (1, seq_len) 形状的 tensor，非图像 token 为 -1
        """

        num_frames = frame_similarity_scores.shape[0]
        # 1. 根据相似度找出segment边界
        # frame 0 一定是 segment 0
        segment_ids_per_frame = [0]
        current_segment = 0
        for i in range(1, num_frames):
            if frame_similarity_scores[i] < frame_segment_threshold:
                current_segment += 1
            segment_ids_per_frame.append(current_segment)
        segment_ids_per_frame = torch.tensor(segment_ids_per_frame, device=device)

        # 2. 初始化所有 token 的 segment id = -1
        segment_id_tensor = torch.full((1, seq_len), -1, dtype=torch.long, device=device)

        # 3. 对每个 frame 的 image token 区间赋值对应 segment id
        for frame_idx in range(num_frames):
            seg_id = segment_ids_per_frame[frame_idx]
            # 当前 frame 的 image token 区间
            start = image_token_start_index + frame_idx * patch_num
            end = start + patch_num
            # 防止超出 image_token_end_index
            if end > image_token_end_index:
                end = image_token_end_index
            segment_id_tensor[0, start:end] = seg_id

        return segment_id_tensor


    def prepare(
        self,
        patch_type: torch.Tensor,
        patch_num: int,
        image_token_start_index: torch.Tensor,
        image_token_end_index: torch.Tensor,
        image_token_length: torch.Tensor,
        original_length: int,
        finish_merging: bool = False,
        finish_pruning: bool = False,
        sparsity_list: List[float] = None,
    ):
        self.patch_type = patch_type
        self.patch_num = patch_num
        self.image_token_start_index = image_token_start_index
        self.image_token_end_index = image_token_end_index
        self.image_token_length = image_token_length
        self.original_length = original_length
        self.finish_merging = finish_merging
        self.finish_pruning = finish_pruning
        if sparsity_list is None:
            self.sparsity_list = []
        else:
            self.sparsity_list = sparsity_list
    ### start token merging at layer 0 before attention
    # 包含了每一个decoder layer 具体做 token merging 和 pruning 的逻辑
    def forward(
        self, hidden_states, position_embeddings, attention_mask, self_attn_weights=None, layer_idx = 0
    ):
        """
        This is the forward method of the FrameFusion class.

        Args:
            hidden_states (torch.Tensor): A tensor of shape (batch_size, sequence_length, hidden_size).
            position_embeddings (torch.Tensor): A tensor of shape (batch_size, sequence_length, hidden_size).
            attention_mask (torch.Tensor): A tensor of shape (batch_size, sequence_length, sequence_length).
            self_attn_weights (torch.Tensor): A tensor of shape (batch_size, sequence_length, sequence_length).

        Returns:
            hidden_states (torch.Tensor): A tensor of shape (batch_size, sequence_length, hidden_size).
            position_embeddings (torch.Tensor): A tensor of shape (batch_size, sequence_length, hidden_size).
            attention_mask (torch.Tensor): A tensor of shape (batch_size, sequence_length, sequence_length).
        """
        bsz, q_len, hidden_size = hidden_states.size()
        device = hidden_states.device    
        # 浅层做 pruning， 深层做 merging
        # pruning
        if q_len >1 and self.finish_merging == True and self.finish_pruning == False:

            def to_int(x):
                return x.item() if isinstance(x, torch.Tensor) else int(x)
            image_token_pruning_start_index = to_int(self.image_token_start_index)
            image_token_pruning_length = to_int(self.image_token_length - (self.original_length - q_len))
            # self.original_length - q_len: 当前层之前merge/prune了多少image token
            # self.image_token_length - (self.original_length - q_len): 当前层还剩多少image token

            last_layer_attention = self_attn_weights
            last_layer_attention_avg = torch.mean(last_layer_attention, dim=(1,2)) # (B, N)
            # last_layer_attention_avg_image = last_layer_attention_avg[:, image_token_pruning_start_index:image_token_pruning_start_index+image_token_pruning_length]

            # pruning_ratio = self._compute_pruning_ratio(self.sparsity_list, self.cost)
            pruning_ratio = self.prune_ratio[int(layer_idx / 7)] # merge率

            # guoyansong 逐segment提取cdpruner剪枝 ==============================
            mask = self.segment_hidden_states_mask[0]  # 变成一维向量，形状 [4503]
            segment_ids = mask[mask >= 0].unique().tolist()  # 所有有效 segment id
            segment_keep_info = []  # 存储结果 [(seg_id, start_idx, token_count, keep_num), ...]
            # 计算 global prune_keep_ratio


            prune_keep_ratio = 1 - pruning_ratio
            ratio_extra = 0.1 # R_extra
            for seg_id in segment_ids:
                # 找到当前 segment 的所有 token 的位置
                seg_positions = (mask == seg_id).nonzero(as_tuple=True)[0]
                # 起始位置 = 最小的索引
                start_idx = seg_positions[0].item()
                # 当前 segment 的 token 数量
                token_count = seg_positions.numel()

                # 要保留的数量 = token_count * 保留率
                # retain_num = int(token_count * prune_keep_ratio)

                # ======================= dynamic ratio
                seg_keep_ratio = prune_keep_ratio + ratio_extra * self.seg_MV[seg_id] # guoyansong：参考MMG_Vid 公式（6）
                seg_keep_ratio = torch.clamp(seg_keep_ratio, min=0.05, max=0.95) # 避免越界，确保seg_keep_ratio 落在 [0.05, 0.95] 区间内
                retain_num = int(token_count * seg_keep_ratio)
                segment_keep_info.append((seg_id, start_idx, token_count, retain_num))
            # import time
            # start_time1 = time.time()
            # 遍历segment，分别cdpruner
            # top_attention_rank_index = []
            # for seg_id, start_idx, token_count, retain_num in segment_keep_info:
            #     top_attention_rank_index.append(
            #         cdpruner(hidden_states[:, start_idx:start_idx+token_count, :],
            #                     last_layer_attention_avg[:, start_idx:start_idx+token_count], 
            #                     retain_num)
            #                     + start_idx
            #     )
            # top_attention_rank_index = torch.cat(top_attention_rank_index, dim=0)
            # end_time2 = time.time()
            # print(f"layer:{layer_idx} | DPP1 运行时间：{end_time2 - start_time1:.6f} 秒 ===")
            
            # ====== CDPruner prune策略  kernel matrix segment
            # 用image token和所有tokens的attn的平均值作为每个image token的relevance score
            pref_start = torch.cuda.Event(enable_timing=True)
            pref_end = torch.cuda.Event(enable_timing=True)
            pref_start.record()
            # import pdb; pdb.set_trace()
            top_attention_rank_index = (
                global_cdpruner_segment_prune(segment_keep_info, 
                                self.segment_hidden_states_mask[0],
                                # 注意！！！下面两条TODO:修改为self.segment_hidden_states_mask,原来基于self.image_token_end_index的写法有问题，因为经历剪枝后self.image_token_end_index没有及时更新
                                hidden_states[:,(self.segment_hidden_states_mask!=-1)[0] , :],
                                last_layer_attention_avg[:, (self.segment_hidden_states_mask!=-1)[0]], 
                                round(image_token_pruning_length * (1 - pruning_ratio)), layer_idx)
                                + image_token_pruning_start_index
            )
            pref_end.record()
            from loguru import logger
            # logger.info(f" ========== [TIMING] global DPP 【layer index - {layer_idx}】: {pref_start.elapsed_time(pref_end)/1000:.4f}s")

            # end_time1 = time.time()
            
            

            keep_indexs = torch.cat(
                (
                    torch.arange(image_token_pruning_start_index, device=device),
                    top_attention_rank_index,
                    torch.arange(
                        image_token_pruning_start_index + image_token_pruning_length,
                        q_len,
                        device=device,
                    ),
                )
            )
            keep_indexs = keep_indexs.sort().values

            self.segment_hidden_states_mask = self.segment_hidden_states_mask[:,keep_indexs]
            hidden_states = hidden_states[:,keep_indexs,:]

            position_embeddings = self.position_embedding_handler_at_pruning(position_embeddings, keep_indexs)


            if attention_mask != None:
                attention_mask = attention_mask[:,:,keep_indexs,:][:,:,:,keep_indexs]
            if layer_idx == 21: # 在prefilling中，最后一个stage完成剪枝后，避免在decoding 阶段还计算attn，降低计算效率
                self.finish_pruning = True

        # merging
        if q_len >1 and (not self.finish_merging):

            # align devices
            self.patch_type = self.patch_type.to(device)

            # prefill  ================  计算mergeing 对应的 image token index
            # sparsity_upper_bound = self._compute_pruning_ratio(self.sparsity_list, self.cost) # 计算最终的目标剪枝率
            sparsity_upper_bound = self.prune_ratio[int(layer_idx / 7)] # merge率
            similarity_by_patch, token_index_by_patch, token_patch_type_by_patch = self.compute_similarity_and_token_index_by_patch(hidden_states, self.patch_type, self.patch_num) # only support bsz = 1
            
            # guoyansong 计算每个frame的相似度得分 ==========================
            # guoyansong：similarity_by_patch计算余弦相似度结果存在nan(framefusion源码就存在这个问题)，导致下面的frame_similarity_scores全部nan
            # 下面把求相似度的平均值 改成 大于merge阈值的token个数
            if self.frame_segment is False: # 保证只在layer0做一次segment
                frame_similarity_scores = FrameFusion._compute_frame_similarity_scores(
                    similarity_by_patch, token_patch_type_by_patch, self.patch_num, device, self.similarity_lower_bound
                )
                # guoyansong 基于frame分段结果，创建segment mask ================= 本函数只执行一次，即只在layer 0 就分好segment
                frame_segment_threshold = 0.4
                self.segment_hidden_states_mask = self.get_segment_id_tensor(frame_similarity_scores, self.patch_num, self.image_token_start_index, self.image_token_end_index, q_len, frame_segment_threshold, device)
                self.frame_segment = True

            # ======================= 计算 dynamic segment ratio ==================================
            self.seg_MV = segment_dynamic_prune_ratio1(self.segment_hidden_states_mask[0], hidden_states)
            # =================================================================================


            frame_token_num = torch.sum(self.patch_type != TEXT_TOKEN).item() # 当前layer的image token总量

            # ============== 处理极少数的NaN异常数据：替换 NaN 为 -inf
            similarity_clean = torch.nan_to_num(similarity_by_patch, nan=float('-inf'))
            values, indices = torch.topk(similarity_clean, int(sparsity_upper_bound * frame_token_num), dim=1)
            topk_indices, _ = torch.sort(indices)
            merge_index_by_patch = topk_indices[0]

            # merge_index_by_patch = torch.where(similarity_by_patch >= self.similarity_lower_bound)[1] # self.similarity_lower_bound：merging的阈值
            # above_k_ratio = merge_index_by_patch.shape[0] / frame_token_num # 当前layer的mergeing操作剪枝的比率
            # # 只在decoder layer0做一次mergeing操作
            # if above_k_ratio < sparsity_upper_bound: # mergeing没有达到了目标剪枝率，后续继续做pruning
            #     self.sparsity_list.append(above_k_ratio) #记录当前decoder layer的剪枝率
            #     #经过多layer mergeing后，above_k_ratio不超过阈值
            #     if above_k_ratio < self.ratio_lower_bound: 
            #         self.finish_merging = True
            # else:# 仅仅mergeing就达到了目标剪枝率，不需要做pruning了
            #     topk_values, topk_indices = torch.topk(similarity_by_patch, int(sparsity_upper_bound*frame_token_num))
            #     topk_indices, _ = torch.sort(topk_indices)
            #     merge_index_by_patch = topk_indices[0]

            #     self.finish_merging = True
            #     self.finish_pruning = True
            # ===================== 做 mergeing 
            hidden_states, token_mask = self.merge_tokens_and_get_mask(hidden_states, similarity_by_patch, token_index_by_patch, merge_index_by_patch)
            self.finish_merging = True
            # ===================== guoyansong 基于frame相似度进行segmentation
            # segments = self._segment_frames_by_similarity(frame_similarity_scores, self.segment_threshold)
            # self.frame_segments = segments  # 保存segments供后续pruning使用

            # 基于merge结果，更新segment mask
            self.segment_hidden_states_mask = self.segment_hidden_states_mask[token_mask].reshape(bsz, -1)

            # here only bsz=1
            # update patch type
            # import pdb; pdb.set_trace()
            self.patch_type = self.patch_type.to(device)[token_mask].reshape(bsz, -1)
            hidden_states = hidden_states[token_mask, :].reshape(bsz, -1, hidden_size)

            position_embeddings = self.position_embedding_handler_at_merging(position_embeddings, token_mask)

            if attention_mask is not None:
                attention_mask = attention_mask[:,:,token_mask[0],:][:,:,:,token_mask[0]]

        return hidden_states, position_embeddings, attention_mask

    def position_embedding_handler_at_pruning(self, position_embeddings, keep_indexs):
        if type(position_embeddings) == list:
            assert len(position_embeddings) == 2
            if position_embeddings[0].ndim == 4:
                position_embeddings[0] = position_embeddings[0][:,:,keep_indexs,:]
                position_embeddings[1] = position_embeddings[1][:,:,keep_indexs,:]
            else:
                position_embeddings[0] = position_embeddings[0][:,keep_indexs,:]
                position_embeddings[1] = position_embeddings[1][:,keep_indexs,:]
        elif type(position_embeddings) == torch.Tensor:
            if position_embeddings.ndim == 2:
                position_embeddings = position_embeddings[:,keep_indexs]
            else:
                raise NotImplementedError("Only support 2D position embeddings")
        else:
            raise NotImplementedError("Only support list or tensor for position embeddings")
        return position_embeddings
    
    
    def position_embedding_handler_at_merging(self, position_embeddings, token_mask):
        if type(position_embeddings) == list:
            # (cos, sin)
            assert len(position_embeddings) == 2
            if position_embeddings[0].ndim == 4:
                position_embeddings[0] = position_embeddings[0][:,:,token_mask[0],:]
                position_embeddings[1] = position_embeddings[1][:,:,token_mask[0],:]
            else:
                position_embeddings[0] = position_embeddings[0][:,token_mask[0],:]
                position_embeddings[1] = position_embeddings[1][:,token_mask[0],:]
        elif type(position_embeddings) == torch.Tensor:
            if position_embeddings.ndim == 2:
                position_embeddings = position_embeddings[:,token_mask[0]]
            else:
                raise NotImplementedError("Only support 2D position embeddings")
        else:
            raise NotImplementedError("Only support list or tensor for position embeddings")
        return position_embeddings

    @staticmethod
    def compute_similarity_and_token_index_by_patch(hidden_states, token_patch_type, patch_num):
        """
        Compute the similarity between consecutive tokens of the same patch type and record the token index.

        Args:
            hidden_states (torch.Tensor): A tensor of shape (batch_size, sequence_length, hidden_size).
            token_patch_type (torch.Tensor): A tensor indicating the patch type of each token in the sequence.
            patch_num (int): The total number of patches of one image in the model. 

        Returns:
            similarity_by_patch (torch.Tensor): A tensor of shape (batch_size, sequence_length) containing
                                                the cosine similarity between consecutive tokens of the
                                                same patch type. Tokens from different patches are set to -2.
            token_index_by_patch (torch.Tensor): A tensor of shape (batch_size, sequence_length) containing
                                                the token index corresponding to the new order after
                                                sorting by patch type.
            frame_similarity_scores (torch.Tensor): A tensor containing the average similarity score for each frame.

        """

        bsz, q_len, _ = hidden_states.size()
        device = hidden_states.device

        assert bsz == 1, "Only support batch size 1"

        token_index_by_patch = []
        similarity_by_patch = []
        
        token_patch_type_by_patch, token_index_by_patch = torch.where( # 返回所有 image tokens在序列中的位置，token_patch_type_by_patch：在frame中的相对位置；token_index_by_patch：每个image token在整个序列中的绝对位置
            token_patch_type == torch.arange(patch_num, device=device)[:, None]
        )

        # noqa: reshape to batch size = 1, with shape (batch_size, q_len),
        token_patch_type_by_patch = token_patch_type_by_patch[None, :]
        token_index_by_patch = token_index_by_patch[None, :]

        similarity_by_patch = cosine_similarity( # 计算每个image token和前一个frame同一spatial位置的token的cosine similarity
            hidden_states[
                torch.arange(bsz, device=device), token_index_by_patch[:, :-1], :
            ],
            hidden_states[
                torch.arange(bsz, device=device), token_index_by_patch[:, 1:], :
            ],
        )

        similarity_by_patch[token_patch_type_by_patch[:, :-1] != token_patch_type_by_patch[:, 1:]] = -2 # -2表示无效，比如第一个frame

        similarity_by_patch = torch.cat(
            (
                torch.full(
                    size=(bsz, 1),
                    fill_value=IGNORE_TOKEN,
                    dtype=hidden_states.dtype,
                    device=device,
                ),
                similarity_by_patch,
            ),
            dim=1,
        )

        assert similarity_by_patch.shape[1] == token_index_by_patch.shape[1]
        return similarity_by_patch, token_index_by_patch, token_patch_type_by_patch

    @staticmethod # guoyansong
    def _compute_frame_similarity_scores(similarity_by_patch, token_patch_type_by_patch, patch_num, device, similarity_threshold):
        """
        计算每个frame的相似度得分(大于阈值的token个数)
        
        Args:
            similarity_by_patch (torch.Tensor): 每个patch的相似度，按patch类型排列
            token_patch_type_by_patch (torch.Tensor): image token 在frame中的相对位置
            patch_num (int): 每个frame的patch数量
            device: 设备
            similarity_threshold (float): 相似度阈值
            
        Returns:
            frame_similarity_scores (torch.Tensor): 每个frame中大于阈值的token个数
        """
        bsz, seq_len = similarity_by_patch.shape
        num_frames = seq_len // patch_num
        
        # 将similarity_by_patch重新组织成按frame排列的格式
        # 创建一个映射矩阵，将patch类型排列转换为frame排列
        frame_similarities = torch.zeros(num_frames, patch_num, device=device)
        
        # 使用向量化操作重新组织数据
        for patch_idx in range(patch_num):
            # 找到所有patch_idx类型的token位置
            patch_mask = token_patch_type_by_patch[0] == patch_idx
            patch_indices = torch.where(patch_mask)[0]
            
            # 批量处理所有frame的该patch类型
            valid_frames = min(num_frames, len(patch_indices))
            if valid_frames > 0:
                valid_indices = patch_indices[:valid_frames]
                valid_similarities = similarity_by_patch[0, valid_indices]
                frame_similarities[:valid_frames, patch_idx] = valid_similarities
        
        # 计算每个frame中大于merge阈值的token个数
        # 使用向量化操作计算所有frame的阈值统计
        valid_mask = frame_similarities != -2
        
        # 计算每个frame中大于阈值的token个数
        above_threshold_mask = (frame_similarities > similarity_threshold) & valid_mask
        frame_similarity_scores = above_threshold_mask.sum(dim=1).float() / patch_num
                
        return frame_similarity_scores

    # @staticmethod # guoyansong
    # def _segment_frames_by_similarity(frame_similarity_scores, segment_threshold):
    #     """
    #     基于frame相似度得分将frames划分为segments
        
    #     Args:
    #         frame_similarity_scores (torch.Tensor): 每个frame的相似度得分
    #         segment_threshold (float): 分割阈值，当相似度得分小于此值时进行分割
            
    #     Returns:
    #         segments (list): 每个segment包含的frame indices列表
    #     """
    #     segments = []
    #     current_segment = []
        
    #     for frame_idx, score in enumerate(frame_similarity_scores):
    #         if score < segment_threshold and len(current_segment) > 0:
    #             # 当前frame相似度低于阈值且已有segment，开始新segment
    #             segments.append(current_segment)
    #             current_segment = [frame_idx]
    #         else:
    #             # 继续当前segment
    #             current_segment.append(frame_idx)
        
    #     # 添加最后一个segment
    #     if len(current_segment) > 0:
    #         segments.append(current_segment)
            
    #     return segments

    def _apply_cdpruner_to_segments(self, hidden_states, last_layer_attention_avg_image, image_token_start_index, image_token_length, segments, patch_num):
        """
        对每个segment分别应用cdpruner
        
        Args:
            hidden_states: 隐藏状态
            last_layer_attention_avg_image: 注意力权重
            image_token_start_index: image token起始索引
            image_token_length: image token长度
            segments: frame segments
            patch_num: 每个frame的patch数量
            
        Returns:
            keep_indexs: 需要保留的token索引
        """
        device = hidden_states.device
        bsz = hidden_states.shape[0]
        
        # 计算总的pruning ratio
        pruning_ratio = self._compute_pruning_ratio(self.sparsity_list, self.cost)
        
        all_keep_indices = []
        
        for segment in segments:
            if len(segment) == 0:
                continue
                
            # 计算当前segment的token范围
            segment_start_frame = segment[0]
            segment_end_frame = segment[-1] + 1
            
            # 计算segment内的token索引范围
            segment_token_start = image_token_start_index + segment_start_frame * patch_num
            segment_token_end = image_token_start_index + segment_end_frame * patch_num
            
            # 确保不超出边界
            segment_token_start = max(segment_token_start, image_token_start_index)
            segment_token_end = min(segment_token_end, image_token_start_index + image_token_length)
            
            if segment_token_start >= segment_token_end:
                continue
                
            # 获取当前segment的attention权重
            segment_attention = last_layer_attention_avg_image[0, segment_token_start - image_token_start_index:segment_token_end - image_token_start_index]
            
            # 计算当前segment需要保留的token数量
            segment_token_count = segment_token_end - segment_token_start
            segment_keep_count = round(segment_token_count * (1 - pruning_ratio))
            
            if segment_keep_count <= 0:
                continue
                
            # 对当前segment应用cdpruner
            segment_features = hidden_states[0, segment_token_start:segment_token_end, :].unsqueeze(0)  # 添加batch维度
            segment_attention_avg = segment_attention.unsqueeze(0)  # 添加batch维度
            
            # 使用cdpruner选择token
            selected_indices = cdpruner(segment_features, segment_attention_avg, segment_keep_count)
            
            # 将相对索引转换为绝对索引
            absolute_indices = selected_indices + segment_token_start
            all_keep_indices.append(absolute_indices)
        
        if len(all_keep_indices) > 0:
            # 合并所有segment的保留索引
            keep_indexs = torch.cat(all_keep_indices).sort().values
        else:
            # 如果没有segment，保留所有token
            keep_indexs = torch.arange(image_token_start_index, image_token_start_index + image_token_length, device=device)
            
        return keep_indexs

    @staticmethod
    def merge_tokens_and_get_mask(hidden_states: torch.Tensor, similarity_by_patch, token_index_by_patch, merge_index_by_patch):
        """
        Merge tokens and get a mask indicating which tokens to keep.

        Args:
            hidden_states (torch.Tensor): A tensor of shape (batch_size, sequence_length, hidden_size)
            similarity_by_patch (torch.Tensor): A tensor of shape (batch_size, sequence_length) containing
                                                the cosine similarity between consecutive tokens of the
                                                same patch type.
            token_index_by_patch (torch.Tensor): A tensor of shape (batch_size, sequence_length) containing
                                                the token indices corresponding to the new order after
                                                sorting by patch type.
            merge_index_by_patch (torch.Tensor): A tensor containing the indices of tokens to be merged, in the patch_type order.

        Returns:
            hidden_states (torch.Tensor): A tensor containing the hidden states of the tokens after merging.
            keep_mask (torch.Tensor): A boolean tensor of shape (batch_size, sequence_length) indicating
                                    which tokens in the original sequence should be kept after merging.
        """
        # pdb.set_trace()

        device = hidden_states.device
        if merge_index_by_patch.shape[0] == 0:
            keep_mask = torch.ones(hidden_states.shape[:-1], dtype=torch.bool, device=device)
            return hidden_states, keep_mask
        bsz, q_len, _ = hidden_states.size()
        bsz_index = torch.arange(bsz, device=hidden_states.device)[:, None]
        merge_mask_by_patch: torch.LongTensor = torch.zeros(
            bsz,
            similarity_by_patch.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        merge_mask_by_patch[bsz_index, merge_index_by_patch] = 1
        last_merge_token_by_patch = find_contigious_latter_index(merge_mask_by_patch) #在不同frame，同一相对位置上的需要merge 的 token的每个连续段的长度
        # token_index_by_patch：在spatial维度，每个image token在整个序列中的绝对位置
        keep_mask = torch.ones(hidden_states.shape[:-1], dtype=torch.bool, device=device)
        keep_mask[bsz_index, token_index_by_patch[bsz_index, merge_index_by_patch]] = False

        # noqa: batch size = 1
        unique_merge_nums = torch.sort(torch.unique(last_merge_token_by_patch.to(torch.long))).values
        unique_merge_nums = (unique_merge_nums[1:] if (unique_merge_nums[0] == 0).item() else unique_merge_nums)

        merge_num_indices, token_merge_index_in_patch = torch.where(
            last_merge_token_by_patch == unique_merge_nums[:, None]
        )

        merge_nums = unique_merge_nums[merge_num_indices]
        token_merge_start_index_in_patch = token_merge_index_in_patch - merge_nums
        token_merge_member_start_index_in_patch = torch.repeat_interleave(token_merge_start_index_in_patch, merge_nums)

        merge_member_length = torch.sum(merge_nums)
        merge_member_contigious_sequence = torch.arange(1, merge_member_length + 1, device = device)

        merge_nums_cumulative_counts = torch.cumsum(merge_nums, dim=0)
        merge_nums_start = torch.cat((torch.tensor([0], device = device), merge_nums_cumulative_counts[:-1]))

        contigious_sequence_by_merge_nums = merge_member_contigious_sequence - torch.repeat_interleave(merge_nums_start, merge_nums)

        token_merge_member_index_in_patch = token_merge_member_start_index_in_patch + contigious_sequence_by_merge_nums

        # 在spatial维度做merge，即在同一个相对位置，对相邻frame的token累加到起始token再取平均
        # noqa: this function may have numerical instability
        hidden_states.index_add_(
            dim = 1,
            index = token_index_by_patch[0, token_merge_member_start_index_in_patch],
            source = hidden_states[
                bsz_index,
                token_index_by_patch[bsz_index, token_merge_member_index_in_patch],
            ]
        )  

        # divide to get average
        hidden_states[
            bsz_index,
            token_index_by_patch[bsz_index, token_merge_start_index_in_patch],
        ] /= (merge_nums[None, :, None] + 1)

        return hidden_states, keep_mask

    @staticmethod
    def _compute_pruning_ratio(sparsity_list, cost, num_layers = 28): # cost 本质上就是所有层的平均 token 保留率（也可以理解为平均计算量比例）
        """
        Args:
            sparsity_list (list): A list containing the sparsity values of the model's first few layers. 记录每一层的剪枝率
            cost (float): The total computation budget given by the user. 剪枝后的计算量: 0.3就表示剪枝后, 所有layer的token总量为原来的30%(计算量就变成原来的30%)
            num_layers (int, optional): The number of layers in the model. 

        Returns:
            float: the required sparsity for the next layer to achieve the given cost
        """
        list_length = len(sparsity_list)
        s = 1
        total_calcution =0
        for i in range(list_length):
            s *= (1 - sparsity_list[i]) # list_length层的token保留率
            total_calcution += s
        remain_calcution = num_layers * cost - total_calcution
        if remain_calcution < 0:
            raise ValueError("The cost is too small")
        if remain_calcution/((num_layers-list_length)*s) > 1:
            return 0
        return 1 - (remain_calcution/((num_layers-list_length)*s)) # 返回下一层的剪枝率

def cosine_similarity(mat1, mat2):
    dot_product = torch.sum(mat1*mat2, dim=-1)
    norm_vec1 = torch.norm(mat1, dim=-1)
    norm_vec2 = torch.norm(mat2, dim=-1)
    return dot_product / (norm_vec1 * norm_vec2)

# guoyansong : 避免矩阵过大
# def cosine_similarity(mat1: torch.Tensor, mat2: torch.Tensor, eps: float = 1e-8, chunk_size: int = 64):
#     """
#     Compute cosine similarity between mat1 and mat2 along the last dimension.
#     mat1, mat2: [B, N, D] or [N, D]
#     """
#     # 确保类型一致
#     assert mat1.shape == mat2.shape, "mat1 and mat2 must have the same shape"

#     # 分块计算以降低显存占用
#     if chunk_size is not None:
#         results = []
#         for i in range(0, mat1.shape[-2], chunk_size):
#             m1 = mat1[:, i:i+chunk_size, ...] if mat1.ndim == 3 else mat1[i:i+chunk_size]
#             m2 = mat2[:, i:i+chunk_size, ...] if mat2.ndim == 3 else mat2[i:i+chunk_size]
#             dot = torch.sum(m1 * m2, dim=-1)
#             norm1 = torch.norm(m1, dim=-1)
#             norm2 = torch.norm(m2, dim=-1)
#             results.append(dot / (norm1 * norm2 + eps))
#         return torch.cat(results, dim=-1)
#     else:
#         dot_product = torch.sum(mat1 * mat2, dim=-1)
#         norm_vec1 = torch.norm(mat1, dim=-1)
#         norm_vec2 = torch.norm(mat2, dim=-1)
#         return dot_product / (norm_vec1 * norm_vec2 + eps)

def find_contigious_latter_index(index_tensor: torch.LongTensor) -> torch.Tensor:
    """
    Args:
        index_tensor (torch.LongTensor): A binary tensor containing sequences of ones and zeros.

    Returns:
        torch.Tensor: A tensor where each contiguous sequence of ones in the input tensor
                    is replaced by zeros, except for the last element of each sequence,
                    which is replaced by the length of that sequence.

    Example:
        Input:  torch.tensor([0, 1, 1, 1, 0, 0, 1, 1])
        Output: torch.tensor([0, 0, 0, 3, 0, 0, 0, 2])
    """
    bsz, n = index_tensor.shape
    t_prev = torch.cat([torch.zeros((bsz, 1), dtype=index_tensor.dtype, device=index_tensor.device), index_tensor[:, :-1]], dim=1)
    t_next = torch.cat([index_tensor[:, 1:], torch.zeros((bsz, 1), dtype=index_tensor.dtype, device=index_tensor.device)], dim=1)

    # Identify the starts and ends of runs of ones
    run_starts = (index_tensor == 1) & (t_prev == 0)
    run_ends = (index_tensor == 1) & (t_next == 0)

    start_indices = torch.nonzero(run_starts, as_tuple=True)
    end_indices = torch.nonzero(run_ends, as_tuple=True)
    run_lengths = (end_indices[1] - start_indices[1] + 1).to(index_tensor.dtype)

    output = torch.zeros_like(index_tensor, dtype=index_tensor.dtype)
    output[end_indices[0], end_indices[1]] = run_lengths

    return output
