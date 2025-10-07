# [CDPruner] Generate index masks using conditional DPP
def cdpruner(hidden_states, self_attn_weights, topk_image_token_num):
    # 预处理特征
    image_features = hidden_states[:, image_token_pruning_start_index:image_token_pruning_start_index+image_token_pruning_length, :]
    # 用image token和所有tokens的attn的平均值作为每个image token的relevance score
    last_layer_attention = self_attn_weights
    last_layer_attention_avg = torch.mean(last_layer_attention, dim=(1,2)) # (B, N)


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
    
    relevance = (-last_layer_attention_avg) # (B, N)
    relevance = (relevance - relevance.min() + 1e-6) / (relevance.max() - relevance.min()) # (B, N)



    # [CDPruner] Construct kernel matrix
    # You can use an additional hyperparameter theta to control the influence of the relevance score.
    # theta = 0.5
    # alpha = theta / (2 * (1 - theta))
    # relevance = torch.exp(alpha * relevance) # (B, N)
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
    import pdb; pdb.set_trace()
    select_idx = torch.sort(select_idx.t()).values # (B, T)
    # index_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
    # index_masks.scatter_(1, select_idx, True)
    
    return select_idx