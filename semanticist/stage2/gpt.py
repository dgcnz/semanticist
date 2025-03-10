# Modified from:
#   VQGAN:    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/transformer/mingpt.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py  
#   nanoGPT:  https://github.com/karpathy/nanoGPT/blob/master/model.py
#   llama:    https://github.com/facebookresearch/llama/blob/main/llama/model.py
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
#   PixArt:   https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
from typing import Optional, List, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from semanticist.stage1.vision_transformer import DropPath
from semanticist.stage2.diffloss import DiffLoss

def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)



#################################################################################
#                      Embedding Layers for Class Labels                        #
#################################################################################
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


#################################################################################
#                                  GPT Model                                    #
#################################################################################
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        multiple_of: int = 256,
        ffn_dropout_p: float = 0.0,
    ):
        super().__init__()
        hidden_dim = 4 * dim
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = find_multiple(hidden_dim, multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.ffn_dropout = nn.Dropout(ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_head: int,
        attn_dropout_p: float = 0.0,
        resid_dropout_p: float = 0.1,
    ):
        super().__init__()
        assert dim % n_head == 0
        self.dim = dim
        self.head_dim = dim // n_head
        self.n_head = n_head

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(dim, dim * 3, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)

    def forward(
        self, x: torch.Tensor, 
        input_pos: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wqkv(x).split([self.dim, self.dim, self.dim], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_head, self.head_dim)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv

        output = F.scaled_dot_product_attention(
            xq, keys, values, 
            attn_mask=mask, 
            is_causal=True if mask is None else False, # is_causal=False is for KV cache
            dropout_p=self.attn_dropout_p if self.training else 0)            
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_head: int,
        multiple_of: int = 256,
        norm_eps: float = 1e-5,
        attn_dropout_p: float = 0.0,
        ffn_dropout_p: float = 0.1,
        resid_dropout_p: float = 0.1,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.attention = Attention(
            dim=dim,
            n_head=n_head,
            attn_dropout_p=attn_dropout_p,
            resid_dropout_p=resid_dropout_p,
        )
        self.feed_forward = FeedForward(
            dim=dim,
            multiple_of=multiple_of,
            ffn_dropout_p=ffn_dropout_p,
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None):
        h = x + self.drop_path(self.attention(self.attention_norm(x), start_pos, mask))
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int = 4096,
        n_layer: int = 32,
        n_head: int = 32,
        attn_dropout_p: float = 0.0,
        resid_dropout_p: float = 0.1,
        ffn_dropout_p: float = 0.1,
        drop_path_rate: float = 0.0,
        num_classes: Union[int, List[int]] = 1000,
        class_dropout_prob: float = 0.1,

        cls_token_num: int = 1,
        num_slots: int = 16,
        slot_dim: int = 256,

        diffloss_d: int = 3,
        diffloss_w: int = 1024,
        num_sampling_steps: str = '100',
        diffusion_batch_mul: int = 4,
        predict_xstart: bool = False,
        use_si: bool = False,
        cond_method: str = "adaln",
        **kwargs,
    ):
        super().__init__()
        
        # Store configuration
        self.dim = dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_classes = num_classes
        self.cls_token_num = cls_token_num
        
        # Initialize embeddings
        self.cls_embedding = LabelEmbedder(num_classes, dim, class_dropout_prob)
        self.z_proj = nn.Linear(slot_dim, dim, bias=True)
        self.z_proj_ln = RMSNorm(dim)
        self.pos_embed_learned = nn.Parameter(torch.zeros(1, num_slots + cls_token_num, dim))

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(n_layer):
            self.layers.append(TransformerBlock(
                dim=dim,
                n_head=n_head,
                ffn_dropout_p=ffn_dropout_p,
                attn_dropout_p=attn_dropout_p,
                resid_dropout_p=resid_dropout_p,
                drop_path=dpr[layer_id],
            ))

        # output layer
        self.norm = RMSNorm(dim)

        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, num_slots, dim))

        # KVCache
        self.max_batch_size = -1
        self.max_seq_length = -1

        self.initialize_weights()

        # Diffusion Loss
        self.diffloss = DiffLoss(
            target_channels=slot_dim,
            z_channels=dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            predict_xstart=predict_xstart,
            use_si=use_si,
            cond_method=cond_method,
        )
        self.diffusion_batch_mul = diffusion_batch_mul

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed_learned, std=0.02)
        nn.init.normal_(self.diffusion_pos_embed_learned, std=0.02)
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(std=0.02)

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        # if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
        #     return
        head_dim = self.dim // self.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.n_head, head_dim, dtype)

        causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)

    def reset_caches(self):
        self.max_seq_length = -1
        self.max_batch_size = -1
        for b in self.layers:
            b.attention.kv_cache = None

    def forward_loss(self, z, target):
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        loss = self.diffloss(z=z, target=target)
        return loss
    
    def forward_cfg(self, h, cfg):
        if cfg > 1.0:
            h_cond, h_uncond = h.chunk(2, dim=0)
            h = h_uncond + cfg * (h_cond - h_uncond)
        return h

    def forward(
        self,
        slots: torch.Tensor,
        cond_idx: torch.Tensor,
        input_pos:  Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        cfg: float = 1.0,
        temperature: float = 1.0
    ):
        if slots is not None and cond_idx is not None: # training or naive inference
            cond_embeddings = self.cls_embedding(cond_idx, train=self.training)
            cond_embeddings = cond_embeddings.expand(-1, self.cls_token_num, -1)
            token_embeddings = self.z_proj(slots)
            token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)
        else:
            if cond_idx is not None: # prefill in inference
                token_embeddings = self.cls_embedding(cond_idx, train=self.training)
                token_embeddings = token_embeddings.expand(-1, self.cls_token_num, -1)
            else: # decode_n_tokens(kv cache) in inference
                token_embeddings = self.z_proj(slots)
            
            bs = token_embeddings.shape[0]
            mask = self.causal_mask[:bs, None, input_pos]
        
        h = token_embeddings
        if self.training:
            h = h + self.pos_embed_learned
        else:
            h = h + self.pos_embed_learned[:, input_pos].view(1, -1, self.dim)
        
        h = self.z_proj_ln(h) # not sure if this is needed

        # transformer blocks
        for layer in self.layers:
            h = layer(h, input_pos, mask)
        
        h = self.norm(h)
        
        if self.training:
            h = h[:, self.cls_token_num - 1 : -1].contiguous()
            h = h + self.diffusion_pos_embed_learned
            loss = self.forward_loss(h, slots.detach())
            return loss
        else:
            h = h[:, -1]
            h = h + self.diffusion_pos_embed_learned[:, input_pos[-1] - self.cls_token_num + 1]
            next_tokens = self.diffloss.sample(h, temperature=temperature, cfg=cfg)
            return next_tokens


    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)



#################################################################################
#                                GPT Configs                                    #
#################################################################################
### text-conditional
def GPT_7B(**kwargs):
    return Transformer(n_layer=32, n_head=32, dim=4096, **kwargs) # 6.6B

def GPT_3B(**kwargs):
    return Transformer(n_layer=24, n_head=32, dim=3200, **kwargs) # 3.1B

def GPT_1B(**kwargs):
    return Transformer(n_layer=22, n_head=32, dim=2048, **kwargs) # 1.2B

### class-conditional
def GPT_XXXL(**kwargs):
    return Transformer(n_layer=48, n_head=40, dim=2560, **kwargs) # 3.9B

def GPT_XXL(**kwargs):
    return Transformer(n_layer=48, n_head=24, dim=1536, **kwargs) # 1.4B

def GPT_XL(**kwargs):
    return Transformer(n_layer=36, n_head=20, dim=1280, **kwargs) # 775M

def GPT_L(**kwargs):
    return Transformer(n_layer=24, n_head=16, dim=1024, **kwargs) # 343M

def GPT_B(**kwargs):
    return Transformer(n_layer=12, n_head=12, dim=768, **kwargs) # 111M
        

GPT_models = {
    'GPT-B': GPT_B, 'GPT-L': GPT_L, 'GPT-XL': GPT_XL, 'GPT-XXL': GPT_XXL, 'GPT-XXXL': GPT_XXXL,
    'GPT-1B': GPT_1B, 'GPT-3B': GPT_3B, 'GPT-7B': GPT_7B, 
}