import numpy as np
import torch
import math
from typing import Any
import os
import torchvision.transforms as T
import matplotlib.pyplot as plt

# meta
TEXT_TOKEN = -1
IGNORE_TOKEN = -2

def get_attr_by_name(obj: Any, name: str) -> Any:
    """
    Get an attribute from an object using a dot notation string.
    e.g., get_attr_by_name(model, "layers.0.self_attn.q_proj") will return model.layers[0].self_attn.q_proj
    """
    levels = name.split('.')
    current = obj
    for level in levels:
        if level.isdigit():
            current = current[int(level)]
        else:
            current = getattr(current, level)
    return current

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os  # 用于创建目录
from matplotlib.colors import LinearSegmentedColormap

def plot_attention_weights(
    layer_idx,
    attn_weight, 
    figsize=(10, 8),
    save_path="/home/hunterj/gys/VidTokenPrune/plots_output",
    q = 0.9 # 绘制前q的attn
):
    """
    绘制注意力权重热力图（保留原始位置，仅将数值前50%的元素置1，其余置0）
    """
    os.makedirs(save_path, exist_ok=True)
    
    # 处理张量（转移到CPU + 转为float32，保留原始形状和位置）
    if attn_weight.device.type != 'cpu':
        attn_weight = attn_weight.cpu()
    modified_attn = attn_weight.detach().float().clone()  # clone()避免修改原张量
    
    # --------------------------
    # 绘制前20%的attn
    # --------------------------
    half_quantile = modified_attn.quantile(q=q)
    modified_attn[modified_attn >= half_quantile] = 1.0  # 前50%置1
    modified_attn[modified_attn < half_quantile] = 0.0   # 后50%置0
    
    # 转为numpy用于绘图（此时每个元素的位置和原始矩阵完全一致）
    attn_np = modified_attn.numpy()
    
    # 后续绘图逻辑不变（基于原始位置的修改后数值）
    q_len = attn_np.shape[0]
    ticks = list([0, 14, 839, 934])
    ticks_name = list(["sys", "video", "prompt"])
    mid_points = [(ticks[i] + ticks[i+1]) / 2 for i in range(len(ticks) - 1)]

    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=figsize)

    # 定义自定义颜色映射：从白色到绿色
    cmap_green = LinearSegmentedColormap.from_list("white_to_green", ["white", "green"])
    ax = sns.heatmap(
        attn_np, 
        annot=False,
        fmt=".2f", 
        cmap=cmap_green, 
        cbar=True,
    )
    ax.set_xticks(mid_points)
    ax.set_xticklabels(ticks_name)
    ax.set_yticks(mid_points)
    ax.set_yticklabels(ticks_name)
    # ✅ 绘制垂直 + 水平分界线 & 边界刻度值
    for t in ticks[1:]:
        # 垂直分界线 + x轴文字
        ax.axvline(t, color='black', linestyle='--', linewidth=1)
        # ax.text(t, -30, str(t), color='black', ha='center', va='top', fontsize=8)
        
        # 水平分界线 + y轴文字
        ax.axhline(t, color='black', linestyle='--', linewidth=1)
        # ax.text(-50, t, str(t), color='black', ha='right', va='center', fontsize=8)
    
    plt.title(f"Attention Weights layer{layer_idx}", fontsize=15)
    plt.xlabel("Key Position", fontsize=12)
    plt.ylabel("Query Position", fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()

    full_save_path = os.path.join(save_path, f"attn-layer{layer_idx}-4frames")
    plt.savefig(full_save_path, dpi=300, bbox_inches="tight")
    print(f"图像已保存至：{full_save_path}")
    # plt.show()

def scaled_dot_product_attention_experiment(layer_idx, query, key, value, num=1, attn_mask=None, dropout_p=0.0,
    is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
        # query = query[:,:,-num:,:]
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).triu(diagonal=S - L + 1)
            attn_bias.masked_fill_(temp_mask, float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)


        attn_weight = query@ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.mean(attn_weight, dim=1).squeeze(0) # torch.Size([q_len, q_len])

        # 调用绘图函数
        plot_attention_weights(
            layer_idx,
            attn_weight,
        )


        return attn_weight

def scaled_dot_product_attention(query, key, value, num=1, attn_mask=None, dropout_p=0.0,
    is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
        query = query[:,:,-num:,:]
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).triu(diagonal=S - L + 1)
            attn_bias.masked_fill_(temp_mask, float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)


        attn_weight = query@ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        if(num > 1): # guoyansong: 这里改成用question做query计算attn weights再取平均
            attn_weight = attn_weight.mean(-2, keepdim=True)
            attn_weight
        attn_weight=attn_weight
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

        return attn_weight

def save_video_frames(video, output_path: str = "local/video_frames"):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    to_pil = T.ToPILImage()
    for i, frame in enumerate(video[0]):
        frame_float = frame.to(torch.float32)
        frame_float = (frame_float + 1) / 2
        frame_float = torch.clamp(frame_float, 0, 1)
        frame_pil = to_pil(frame_float)
        frame_pil.save(os.path.join(output_path, f"frame_{i}.png"))

def save_video_frames_subfigures(video, output_path: str = "local/video_frames.jpg"):
    """
    Save the video frames as subfigures in a single image.
    """
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
        
    num_frames = len(video[0])
    rows = int(np.sqrt(num_frames))
    cols = int(np.ceil(num_frames / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten()
    
    to_pil = T.ToPILImage()
    for i, frame in enumerate(video[0]):
        frame_float = frame.to(torch.float32)
        frame_float = (frame_float + 1) / 2
        frame_float = torch.clamp(frame_float, 0, 1)
        frame_pil = to_pil(frame_float)
        
        axes[i].imshow(frame_pil)
        axes[i].axis('off')
        axes[i].set_title(f'Frame {i}')
    
    # Hide empty subplots
    for i in range(num_frames, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
