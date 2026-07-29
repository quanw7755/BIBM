"""
  @Author: 王权
  @FileName: enhance_controller.py
  @DateTime: 2026/5/9 22:48
  @SoftWare: PyCharm
"""
import math
import torch
import torch.nn as nn


def _safe_logit(x, eps=1e-6):
    x = min(max(float(x), eps), 1.0 - eps)
    return math.log(x / (1.0 - x))


class TokenEnhanceController(nn.Module):
    def __init__(
        self,
        dim,
        args,
        structure_aggregator_cls,
        hrami_cls,
        freq_cls,
    ):
        super().__init__()

        self.use_freq = bool(args.use_freq)
        self.use_hrami = bool(args.use_hrami)
        self.enhance_order = args.enhance_order
        self.learnable_module_gates = bool(args.learnable_module_gates)

        self.structure_aggregator = structure_aggregator_cls(
            scales=(1, 3, 5),
            learnable_fusion=True
        )

        if self.use_freq:
            self.freq_enhancer = freq_cls(
                dim=dim,
                reduction=4,
                cutoff=args.freq_cutoff,
                freq_mode=getattr(args, "freq_mode", "both")
            )

            # 如果你已经把 FrequencyTokenEnhancer 里的 gamma 改成可传入，
            # 可以在它内部使用 args.freq_init_gamma。
        else:
            self.freq_enhancer = None

        if self.use_hrami:
            self.hrami_ad = hrami_cls(
                dim=dim,
                region_grids=tuple(args.hrami_grids),
                reduction=8,
                cls_fusion=False,
                init_value=args.hrami_init_value,
            )
        else:
            self.hrami_ad = None

        if self.learnable_module_gates:
            self.freq_gate_logit = nn.Parameter(
                torch.tensor(_safe_logit(args.freq_strength))
            )
            self.hrami_gate_logit = nn.Parameter(
                torch.tensor(_safe_logit(args.hrami_strength))
            )
        else:
            self.register_buffer(
                "freq_gate_value",
                torch.tensor(float(args.freq_strength))
            )
            self.register_buffer(
                "hrami_gate_value",
                torch.tensor(float(args.hrami_strength))
            )

    def _freq_strength(self):
        if self.learnable_module_gates:
            return torch.sigmoid(self.freq_gate_logit)
        return self.freq_gate_value

    def _hrami_strength(self):
        if self.learnable_module_gates:
            return torch.sigmoid(self.hrami_gate_logit)
        return self.hrami_gate_value

    def _blend_token(self, old_token, new_token, strength):
        """
        old_token: [B, 1+L, C]
        new_token: [B, 1+L, C]

        返回:
            old + strength * (new - old)
        """
        return old_token + strength * (new_token - old_token)

    def _apply_freq(self, tokens):
        if not self.use_freq or self.freq_enhancer is None:
            return tokens

        strength = self._freq_strength()
        enhanced = []

        for tok in tokens:
            new_tok = self.freq_enhancer(tok)
            new_tok = self._blend_token(tok, new_tok, strength)
            enhanced.append(new_tok)

        return enhanced

    def _apply_hrami(self, tokens):
        if not self.use_hrami or self.hrami_ad is None:
            return tokens

        strength = self._hrami_strength()

        new_tokens = self.hrami_ad(tokens)
        enhanced = [
            self._blend_token(old, new, strength)
            for old, new in zip(tokens, new_tokens)
        ]

        return enhanced

    def forward(self, raw_tokens):
        """
        raw_tokens:
            list of token tensors from CLIP visual encoder.
            每个 Tensor: [B, 1+L, C]

        return:
            all enhanced tokens, list of [B, 1+L, C]
        """
        all_tokens = []

        for img_token in raw_tokens:
            # 1. 每一层 token 先做多尺度结构聚合
            ms_tokens = self.structure_aggregator(img_token)

            # 2. 中控决定 HRAMI / Frequency 的顺序
            if self.enhance_order == "freq_hrami":
                ms_tokens = self._apply_freq(ms_tokens)
                ms_tokens = self._apply_hrami(ms_tokens)

            elif self.enhance_order == "hrami_freq":
                ms_tokens = self._apply_hrami(ms_tokens)
                ms_tokens = self._apply_freq(ms_tokens)

            else:
                raise ValueError(f"Unknown enhance_order: {self.enhance_order}")

            all_tokens.extend(ms_tokens)

        return all_tokens
