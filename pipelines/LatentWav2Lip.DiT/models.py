# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


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
        embeddings = self.embedding_table(labels)
        return embeddings

class AudioEmbedder(nn.Module):
    """
    Embeds audio features into vector representations.
    """
    def __init__(self, hidden_size, audio_embedding_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(audio_embedding_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, a):
        a_emb = self.mlp(a)
        return a_emb

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class JointAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            # qk_norm: bool = False,
            i_proj_drop: float = 0.,
            a_proj_drop: float = 0.,
            last_block: bool = False,
            # norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # self.scale = self.head_dim ** -0.5
        
        self.qkv_x = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_i = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_a = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        # self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        
        # self.attn_drop = nn.Dropout(attn_drop)
        self.x_proj = nn.Linear(dim, dim)
        self.x_proj_drop = nn.Dropout(i_proj_drop)
                
        self.last_block = last_block
        if not last_block:
            self.i_proj = nn.Linear(dim, dim)
            self.i_proj_drop = nn.Dropout(i_proj_drop)
            self.a_proj = nn.Linear(dim, dim)
            self.a_proj_drop = nn.Dropout(a_proj_drop)

    def forward(self, x: torch.Tensor, i: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        B, N_x, C = x.shape
        x_qkv = self.qkv_x(x).reshape(B, N_x, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        x_q, x_k, x_v = x_qkv.unbind(0)
        
        B, N_i, C = i.shape
        i_qkv = self.qkv_i(i).reshape(B, N_i, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        i_q, i_k, i_v = i_qkv.unbind(0)
        
        B, N_a, C = a.shape
        a_qkv = self.qkv_a(a).reshape(B, N_a, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        a_q, a_k, a_v = a_qkv.unbind(0)
        
        # concat i and a        
        q = torch.cat([x_q, i_q, a_q], dim=2)
        k = torch.cat([x_k, i_k, a_k], dim=2)
        v = torch.cat([x_v, i_v, a_v], dim=2)
        
        # q, k = self.q_norm(q), self.k_norm(k)

        combined = nn.functional.scaled_dot_product_attention(
            q, k, v,
        )

        combined = combined.transpose(1, 2).reshape(B, N_x + N_i + N_a, C)
        x, i, a = combined.split([N_x, N_i, N_a], dim=1)
        
        x = self.x_proj(x)
        
        if not self.last_block:
            i = self.i_proj(i)
            i = self.i_proj_drop(i)
            a = self.a_proj(a)
            a = self.a_proj_drop(a)
        
        return x, i, a
    
class MMDiTBlock(nn.Module):
    """
    A MMDiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, syncnet_T, hidden_size, num_heads, mlp_ratio=4.0, last_block=False, temporal_attention=False, **block_kwargs):
        super().__init__()
        self.syncnet_T = syncnet_T
        self.last_block = last_block
        self.temporal_attention = temporal_attention
        
        # ==== TempoAttention Block ====
        if self.temporal_attention:
            self.x_norm_tmp = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.i_norm_tmp = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.a_norm_tmp = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.x_tmp_att = Attention(dim=hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
            self.i_tmp_att = Attention(dim=hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
            self.a_tmp_att = Attention(dim=hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        
        # ==== Attention Block ====
        self.x_norm_att = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.i_norm_att = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.a_norm_att = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.joint_attn = JointAttention(hidden_size, num_heads=num_heads, qkv_bias=True, last_block=last_block, **block_kwargs)
        
        # ==== MLP Block ====
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.x_norm_mlp = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        x_approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.x_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=x_approx_gelu, drop=0)
        
        if not self.last_block:
            self.i_norm_mlp = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            i_approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.i_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=i_approx_gelu, drop=0)
            self.a_norm_mlp = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            a_approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.a_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=a_approx_gelu, drop=0)
        
        self.x_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.i_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.a_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        
        if self.temporal_attention:
            self.x_adaLN_modulation_tmp = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size, bias=True))
            self.i_adaLN_modulation_tmp = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size, bias=True))
            self.a_adaLN_modulation_tmp = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size, bias=True))
        
    def reshape_for_temporal(self, x):
        total_batch_size, tokens, hidden_size = x.shape
        batch_size = total_batch_size // self.syncnet_T
        x = x.view(batch_size, self.syncnet_T, tokens, hidden_size)
        x = x.permute(0, 2, 1, 3) # (batch_size, tokens, syncnet_T, hidden_size)
        x = x.reshape(batch_size * tokens, self.syncnet_T, hidden_size)
        return x
    
    def reshape_to_original(self, x, batch_size):
        total_batch_size, T, hidden_size = x.shape
        tokens = total_batch_size // batch_size        
        x = x.view(batch_size, tokens, T, hidden_size)
        x = x.permute(0, 2, 1, 3) # (batch_size, T, tokens, hidden_size)
        x = x.reshape(batch_size * T, tokens, hidden_size)
        return x

    def forward(self, x, i, a, c):
        x_shift_msa, x_scale_msa, x_gate_msa, x_shift_mlp, x_scale_mlp, x_gate_mlp = self.x_adaLN_modulation(c).chunk(6, dim=1)
        i_shift_msa, i_scale_msa, i_gate_msa, i_shift_mlp, i_scale_mlp, i_gate_mlp = self.i_adaLN_modulation(c).chunk(6, dim=1)
        a_shift_msa, a_scale_msa, a_gate_msa, a_shift_mlp, a_scale_mlp, a_gate_mlp = self.a_adaLN_modulation(c).chunk(6, dim=1)
        
        # ==== Tempo Attention Block ====
        if self.temporal_attention:
            x_shift_tmp, x_scale_tmp, x_gate_tmp = self.x_adaLN_modulation_tmp(c).chunk(3, dim=1)
            i_shift_tmp, i_scale_tmp, i_gate_tmp = self.i_adaLN_modulation_tmp(c).chunk(3, dim=1)
            a_shift_tmp, a_scale_tmp, a_gate_tmp = self.a_adaLN_modulation_tmp(c).chunk(3, dim=1)
            batch_size = x.shape[0] // self.syncnet_T
            x = x + x_gate_tmp.unsqueeze(1) * self.reshape_to_original(self.x_tmp_att(self.reshape_for_temporal(modulate(self.x_norm_tmp(x), x_shift_tmp, x_scale_tmp))), batch_size)
            i = i + i_gate_tmp.unsqueeze(1) * self.reshape_to_original(self.i_tmp_att(self.reshape_for_temporal(modulate(self.i_norm_tmp(i), i_shift_tmp, i_scale_tmp))), batch_size)
            a = a + a_gate_tmp.unsqueeze(1) * self.reshape_to_original(self.a_tmp_att(self.reshape_for_temporal(modulate(self.a_norm_tmp(a), a_shift_tmp, a_scale_tmp))), batch_size)
            
        # ==== Joint Attention Block ====
        att_x, att_i, att_a = self.joint_attn(modulate(self.x_norm_att(x), x_shift_msa, x_scale_msa), modulate(self.i_norm_att(i), i_shift_msa, i_scale_msa), modulate(self.a_norm_att(a), a_shift_msa, a_scale_msa))
        x = x + x_gate_msa.unsqueeze(1) * att_x
        i = i + i_gate_msa.unsqueeze(1) * att_i
        a = a + a_gate_msa.unsqueeze(1) * att_a
                
        # ==== MLP Block ====
        x = x + x_gate_mlp.unsqueeze(1) * self.x_mlp(modulate(self.x_norm_mlp(x), x_shift_mlp, x_scale_mlp))
        if not self.last_block:
            i = i + i_gate_mlp.unsqueeze(1) * self.i_mlp(modulate(self.i_norm_mlp(i), i_shift_mlp, i_scale_mlp))
            a = a + a_gate_mlp.unsqueeze(1) * self.a_mlp(modulate(self.a_norm_mlp(a), a_shift_mlp, a_scale_mlp))
        return x, i, a


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        syncnet_T=5,
        input_size=32,
        patch_size=2,
        in_channels=4,
        out_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        # class_dropout_prob=0.1,
        audio_dropout_prob=0.1,
        # num_classes=1000,
        audio_embedding_dim=384, # from whisper
        learn_sigma=True,
        temporal_attention=False,
    ):
        super().__init__()
        self.syncnet_T = syncnet_T
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = out_channels * 2 if learn_sigma else out_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.audio_dropout_prob = audio_dropout_prob
        self.temporal_attention = temporal_attention

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        # self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.a_embedder = AudioEmbedder(hidden_size, audio_embedding_dim)
        num_patches = self.x_embedder.num_patches
        
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        
        if self.temporal_attention:
            # Will use fixed sin-cos embedding:
            self.temporal_pos_embed = nn.Parameter(torch.zeros(1, syncnet_T, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            MMDiTBlock(syncnet_T, hidden_size, num_heads, mlp_ratio=mlp_ratio, temporal_attention=temporal_attention) for _ in range(depth - 1)
        ])
        self.blocks.append(MMDiTBlock(syncnet_T, hidden_size, num_heads, mlp_ratio=mlp_ratio, last_block=True, temporal_attention=temporal_attention))
        
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        if self.temporal_attention:
            # Initialize (and freeze) temporal_pos_embed by sin-cos embedding:
            temporal_pos_embed = get_1d_sincos_pos_embed(self.temporal_pos_embed.shape[-1], self.syncnet_T)
            self.temporal_pos_embed.data.copy_(torch.from_numpy(temporal_pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.x_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.x_adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.i_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.i_adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.a_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.a_adaLN_modulation[-1].bias, 0)
            
            if block.temporal_attention:
                nn.init.constant_(block.x_adaLN_modulation_tmp[-1].weight, 0)
                nn.init.constant_(block.x_adaLN_modulation_tmp[-1].bias, 0)
                nn.init.constant_(block.i_adaLN_modulation_tmp[-1].weight, 0)
                nn.init.constant_(block.i_adaLN_modulation_tmp[-1].bias, 0)
                nn.init.constant_(block.a_adaLN_modulation_tmp[-1].weight, 0)
                nn.init.constant_(block.a_adaLN_modulation_tmp[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1], f"Invalid number of patches: {h} x {w} != {x.shape[1]}"

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def add_temporal_pos_embed(self, tensor):
        """
        tensor: (N, T, D)
        """
        N, T, D = tensor.shape
        tensor = tensor.view(N // self.syncnet_T, self.syncnet_T, T, D)
        tensor = tensor.permute(0, 2, 1, 3)
        tensor = tensor.reshape(-1, T, self.syncnet_T, D)
        tensor = tensor + self.temporal_pos_embed
        tensor = tensor.view(N // self.syncnet_T, T, self.syncnet_T, D)
        tensor = tensor.permute(0, 2, 1, 3)
        tensor = tensor.reshape(-1, T, D)
        return tensor
        
    def forward(self, x, t, i, a):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        use_dropout = self.audio_dropout_prob > 0
        if self.training and use_dropout:
            need_dropout = torch.rand(a.shape[0], device=a.device) < self.audio_dropout_prob
            need_dropout = need_dropout[:, None, None]
            a = torch.where(need_dropout, torch.zeros_like(a), a)
        
        B, C, H, W = i.shape
        i = i.view(B * 2, C // 2, H, W)
        i = self.x_embedder(i) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        i = i.reshape(B, -1, i.shape[-1])
        
        if self.temporal_attention:
            # add temporal positional embedding to i
            i = self.add_temporal_pos_embed(i)
            
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2  
        
        if self.temporal_attention:
            # add temporal positional embedding to x
            x = self.add_temporal_pos_embed(x)
        
        a = self.a_embedder(a)                   # (N, T_a, D)
        
        if self.temporal_attention:
            # add temporal positional embedding to a
            a = self.add_temporal_pos_embed(a)
        
        a_c = torch.mean(a, dim=1)               # (N, D), pooling over time
        t = self.t_embedder(t)                   # (N, D)
        c = t + a_c                              # (N, D)
        for block in self.blocks:
            x, i, a = block(x, i, a, c)          # (N, T, D)
        x = self.final_layer(x, c)               # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
            
        return x

    def forward_with_cfg(self, x, t, i, a, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        # print("combined.shape", combined.shape, "t.shape", t.shape, "i.shape", i.shape, "a.shape", a.shape, "x.shape", x.shape)
        model_out = self.forward(combined, t, i, a)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_1d_sincos_pos_embed(embed_dim, num_positions):
    """
    num_positions: int of the sequence length (e.g., number of frames)
    return:
    pos_embed: [num_positions, embed_dim]
    """
    positions = np.arange(num_positions, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, positions)
    return pos_embed


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
