from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HRAMI_AD(nn.Module):
    """
    Hierarchical Region-Aware Multi-scale Interaction for Anomaly Detection

    输入:
        ms_tokens: list of Tensor
            每个元素形状: [B, 1+L, C]
            第一个 token 为 cls token，后面是 patch tokens

    输出:
        enhanced_tokens: list of Tensor
            每个元素仍为 [B, 1+L, C]
    """
    def __init__(
        self,
        dim,
        region_grids=(2, 4, 6),
        reduction=8,
        cls_fusion=True,
        init_value=0.10,
    ):
        super().__init__()
        self.dim = dim
        self.region_grids = region_grids
        self.cls_fusion = cls_fusion

        hidden_dim = max(dim // reduction, 64)

        self.token_norm = nn.LayerNorm(dim)

        # 动态尺度权重
        self.scale_score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

        # 轻量投影，避免 4C 大拼接太重
        self.local_proj = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.fused_proj = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.region_proj = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.disc_proj = nn.Conv2d(1, hidden_dim, kernel_size=1, bias=False)

        self.mix_refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
        )

        nn.init.zeros_(self.mix_refine[-1].weight)

        self.region_gate = nn.Sequential(
            nn.Conv2d(hidden_dim + 1, hidden_dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        nn.init.constant_(self.region_gate[2].bias, -2.0)

        self.patch_scale = nn.Parameter(torch.ones(1, dim, 1, 1) * init_value)

        if self.cls_fusion:
            self.cls_mlp = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim)
            )
            self.cls_scale = nn.Parameter(torch.ones(1, 1, dim) * init_value)

    def _tokens_to_map(self, patch_tokens):
        """
        patch_tokens: [B, L, C]
        return: [B, C, H, W]
        """
        b, l, c = patch_tokens.shape
        h = int(math.sqrt(l))
        w = h
        assert h * w == l, f"[HRAMI_AD] patch token length {l} is not square."
        feat_map = patch_tokens.transpose(1, 2).reshape(b, c, h, w)
        return feat_map

    def _build_region_context(self, feat):
        """
        feat: [B, C, H, W]
        通过多个 region grid 建立区域上下文
        """
        h, w = feat.shape[-2], feat.shape[-1]
        region_ctxs = []
        for g in self.region_grids:
            pooled = F.adaptive_avg_pool2d(feat, output_size=(g, g))
            up = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False)
            region_ctxs.append(up)

        if len(region_ctxs) == 1:
            return region_ctxs[0]
        return torch.stack(region_ctxs, dim=0).mean(dim=0)

    def forward(self, ms_tokens):
        if not isinstance(ms_tokens, (list, tuple)):
            raise TypeError("[HRAMI_AD] ms_tokens should be list/tuple.")

        if len(ms_tokens) == 0:
            return ms_tokens

        cls_tokens = []
        feat_maps = []
        raw_feat_maps = []
        scale_descs = []

        # 1) token -> feature map
        # raw feature 用于最终 residual
        # norm feature 用于计算 HRAMI 增强信息
        for tok in ms_tokens:
            cls_tok = tok[:, :1, :]
            patch_raw = tok[:, 1:, :]
            patch_norm = self.token_norm(patch_raw)

            feat_raw = self._tokens_to_map(patch_raw)
            feat = self._tokens_to_map(patch_norm)

            cls_tokens.append(cls_tok)
            raw_feat_maps.append(feat_raw)
            feat_maps.append(feat)

            desc = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            scale_descs.append(desc)

        # 2) 动态尺度权重
        scale_scores = [self.scale_score(desc) for desc in scale_descs]
        scale_scores = torch.stack(scale_scores, dim=1)
        scale_weights = torch.softmax(scale_scores, dim=1)

        # 3) 融合跨尺度特征
        fused_feat = 0
        for s, feat in enumerate(feat_maps):
            w = scale_weights[:, s].view(feat.size(0), 1, 1, 1)
            fused_feat = fused_feat + w * feat

        enhanced_tokens = []

        # 4) 每个尺度做区域增强
        for s, feat in enumerate(feat_maps):
            feat_raw = raw_feat_maps[s]

            region_ctx = self._build_region_context(feat)
            discrepancy_map = torch.abs(feat - fused_feat).mean(dim=1, keepdim=True)

            mix_hidden = (
                    self.local_proj(feat) +
                    self.fused_proj(fused_feat) +
                    self.region_proj(region_ctx) +
                    self.disc_proj(discrepancy_map)
            )
            mix_hidden = F.gelu(mix_hidden)

            mix_feat = self.mix_refine(mix_hidden)

            gate = self.region_gate(
                torch.cat([mix_hidden, discrepancy_map], dim=1)
            )

            # 关键：这里必须用 feat_raw，而不是 feat
            enhanced_feat = feat_raw + self.patch_scale * gate * mix_feat

            patch_tokens = enhanced_feat.flatten(2).transpose(1, 2)

            if self.cls_fusion:
                summary = F.adaptive_avg_pool2d(enhanced_feat, 1).flatten(1)
                cls_delta = self.cls_mlp(summary).unsqueeze(1)
                new_cls = cls_tokens[s] + self.cls_scale * cls_delta
            else:
                new_cls = cls_tokens[s]

            enhanced_tok = torch.cat([new_cls, patch_tokens], dim=1)
            enhanced_tokens.append(enhanced_tok)

        return enhanced_tokens


class FrequencyTokenEnhancer(nn.Module):
    """
    Frequency-aware token enhancement for CLIP patch tokens.

    Input:
        x: [B, N, C], where N = 1 + H * W
           x[:, 0:1, :] is cls token
           x[:, 1:, :] are patch tokens

    Output:
        enhanced token with the same shape [B, N, C]
    """
    def __init__(self, dim, reduction=4, cutoff=0.25, init_gamma=1e-3, freq_mode='both'):
        super().__init__()
        self.cutoff = cutoff
        self.freq_mode = freq_mode
        hidden_dim = max(dim // reduction, 16)

        # 用通道门控自适应控制低频/高频成分
        self.freq_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim * 2, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # 轻量空间细化，强调局部异常纹理
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        )

        # 零初始化残差系数，保证一开始不破坏原始 CLIP 表征
        self.gamma = nn.Parameter(torch.zeros(1))

        self.norm = nn.LayerNorm(dim)

    def build_freq_masks(self, h, w, device):
        fy = torch.fft.fftfreq(h, device=device).view(h, 1)
        fx = torch.fft.rfftfreq(w, device=device).view(1, w // 2 + 1)

        radius = torch.sqrt(fx ** 2 + fy ** 2)

        low_mask = (radius <= self.cutoff).float()
        high_mask = 1.0 - low_mask

        low_mask = low_mask.view(1, 1, h, w // 2 + 1)
        high_mask = high_mask.view(1, 1, h, w // 2 + 1)

        return low_mask, high_mask

    def forward(self, x):
        dtype = x.dtype
        b, n, c = x.shape

        cls_token = x[:, :1, :]
        patch_token = x[:, 1:, :]

        l = patch_token.shape[1]
        h = w = int(math.sqrt(l))

        if h * w != l:
            raise ValueError(
                f"Patch token length {l} is not a square number, "
                f"cannot reshape to 2D feature map."
            )

        # [B, L, C] -> [B, C, H, W]
        feat = patch_token.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        # torch.fft 对 fp16 支持不稳定，这里转 float 做频域操作
        feat_float = feat.float()

        freq = torch.fft.rfft2(feat_float, norm="ortho")
        low_mask, high_mask = self.build_freq_masks(h, w, feat.device)

        low_freq = torch.fft.irfft2(freq * low_mask, s=(h, w), norm="ortho")
        high_freq = torch.fft.irfft2(freq * high_mask, s=(h, w), norm="ortho")

        gates = self.freq_gate(feat_float)
        low_gate, high_gate = gates.chunk(2, dim=1)

        if self.freq_mode == 'both':
            freq_enhanced = low_gate * low_freq + high_gate * high_freq

        elif self.freq_mode == 'low_only':
            freq_enhanced = low_freq

        elif self.freq_mode == 'high_only':
            freq_enhanced = high_freq

        elif self.freq_mode == 'wo_low':
            freq_enhanced = high_gate * high_freq

        elif self.freq_mode == 'wo_high':
            freq_enhanced = low_gate * low_freq

        else:
            raise ValueError(f"Unknown freq_mode: {self.freq_mode}")

        freq_enhanced = self.spatial_refine(freq_enhanced)

        out = feat_float + self.gamma * freq_enhanced
        out = out.to(dtype)

        # [B, C, H, W] -> [B, L, C]
        out = out.permute(0, 2, 3, 1).reshape(b, l, c)

        out = torch.cat([cls_token, out], dim=1)
        out = self.norm(out)

        return out


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        if dim_out is None:
            dim_out = dim
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, out_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        out_dim = default(out_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, out_dim, n_heads=8, d_head=64, dropout=0.):
        super().__init__()
        self.attn = CrossAttention(query_dim=dim, out_dim=out_dim, heads=n_heads, dim_head=d_head,
                                   dropout=dropout)  # is a self-attention
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.attn(self.norm(x))
        return x


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Adaptor(nn.Module):
    def __init__(self, inplanes=1024, outplanes=None):
        super(Adaptor, self).__init__()
        outplanes = default(outplanes, inplanes)
        self.attention = BasicTransformerBlock(dim=inplanes, out_dim=outplanes)

    def forward(self, img_token):
        return self.attention(img_token)


