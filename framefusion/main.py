from typing import List
import torch
from torch import nn
import pdb

TEXT_TOKEN = -1
IGNORE_TOKEN = -2

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

class FrameFusion(nn.Module):
    def __init__(self, cost=0.3, similarity_lower_bound=0.6, ratio_lower_bound=0.1):
        super(FrameFusion, self).__init__()
        self.cost = cost
        self.similarity_lower_bound = similarity_lower_bound
        self.ratio_lower_bound = ratio_lower_bound

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
        self, hidden_states, position_embeddings, attention_mask, self_attn_weights=None
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
            last_layer_attention_avg_image = last_layer_attention_avg[:, image_token_pruning_start_index:image_token_pruning_start_index+image_token_pruning_length]

            pruning_ratio = self._compute_pruning_ratio(self.sparsity_list, self.cost)

            # ====== FrameFusion prune策略
            # top_attention_rank_index = (
            #     last_layer_attention_avg_image[0].topk( # 0表示batch，这里batch=1，所以是去掉batch这个维度
            #         round(image_token_pruning_length * (1 - pruning_ratio))
            #     ).indices
            #     + image_token_pruning_start_index
            # )
            # round(image_token_pruning_length * (1 - pruning_ratio)): 本轮剪枝后，保留的image token数量

            # ====== CDPruner prune策略
            # 用image token和所有tokens的attn的平均值作为每个image token的relevance score
            top_attention_rank_index = (
                cdpruner(hidden_states[:, image_token_pruning_start_index:image_token_pruning_start_index+image_token_pruning_length, :],
                                last_layer_attention_avg_image, 
                                round(image_token_pruning_length * (1 - pruning_ratio)))
                                + image_token_pruning_start_index
            )
            # pdb.set_trace()
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

            hidden_states = hidden_states[:,keep_indexs,:] 

            position_embeddings = self.position_embedding_handler_at_pruning(position_embeddings, keep_indexs)


            if attention_mask != None:
                attention_mask = attention_mask[:,:,keep_indexs,:][:,:,:,keep_indexs]
            self.finish_pruning = True

        # merging
        if q_len >1 and (not self.finish_merging):
            # align devices
            self.patch_type = self.patch_type.to(device)

            # prefill
            sparsity_upper_bound = self._compute_pruning_ratio(self.sparsity_list, self.cost)
            similarity_by_patch, token_index_by_patch = self.compute_similarity_and_token_index_by_patch(hidden_states, self.patch_type, self.patch_num) # only support bsz = 1

            frame_token_num = torch.sum(self.patch_type != TEXT_TOKEN).item()
            merge_index_by_patch = torch.where(similarity_by_patch >= self.similarity_lower_bound)[1]
            above_k_ratio = merge_index_by_patch.shape[0] / frame_token_num

            if above_k_ratio < sparsity_upper_bound:
                self.sparsity_list.append(above_k_ratio)

                if above_k_ratio < self.ratio_lower_bound:
                    self.finish_merging = True
            else:
                topk_values, topk_indices = torch.topk(similarity_by_patch, int(sparsity_upper_bound*frame_token_num))
                topk_indices, _ = torch.sort(topk_indices)
                merge_index_by_patch = topk_indices[0]

                self.finish_merging = True
                self.finish_pruning = True

            hidden_states, token_mask = self.merge_tokens_and_get_mask(hidden_states, similarity_by_patch, token_index_by_patch, merge_index_by_patch)
            # here only bsz=1
            # update patch type
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

        """

        bsz, q_len, _ = hidden_states.size()
        device = hidden_states.device

        assert bsz == 1, "Only support batch size 1"

        token_index_by_patch = []
        similarity_by_patch = []

        token_patch_type_by_patch, token_index_by_patch = torch.where(
            token_patch_type == torch.arange(patch_num, device=device)[:, None]
        )

        # noqa: reshape to batch size = 1, with shape (batch_size, q_len),
        token_patch_type_by_patch = token_patch_type_by_patch[None, :]
        token_index_by_patch = token_index_by_patch[None, :]

        similarity_by_patch = cosine_similarity(
            hidden_states[
                torch.arange(bsz, device=device), token_index_by_patch[:, :-1], :
            ],
            hidden_states[
                torch.arange(bsz, device=device), token_index_by_patch[:, 1:], :
            ],
        )

        similarity_by_patch[token_patch_type_by_patch[:, :-1] != token_patch_type_by_patch[:, 1:]] = -2

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
        return similarity_by_patch, token_index_by_patch

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
        last_merge_token_by_patch = find_contigious_latter_index(merge_mask_by_patch)

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
    def _compute_pruning_ratio(sparsity_list, cost, num_layers = 28):
        """
        Args:
            sparsity_list (list): A list containing the sparsity values of the model's first few layers.
            cost (float): The total computation budget given by the user.
            num_layers (int, optional): The number of layers in the model. 

        Returns:
            float: the required sparsity for the next layer to achieve the given cost
        """
        list_length = len(sparsity_list)
        s = 1
        total_calcution =0
        for i in range(list_length):
            s *= (1 - sparsity_list[i])
            total_calcution += s
        remain_calcution = num_layers * cost - total_calcution
        if remain_calcution < 0:
            raise ValueError("The cost is too small")
        if remain_calcution/((num_layers-list_length)*s) > 1:
            return 0
        return 1 - (remain_calcution/((num_layers-list_length)*s))    

def cosine_similarity(mat1, mat2):
    dot_product = torch.sum(mat1*mat2, dim=-1)
    norm_vec1 = torch.norm(mat1, dim=-1)
    norm_vec2 = torch.norm(mat2, dim=-1)
    return dot_product / (norm_vec1 * norm_vec2)

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
