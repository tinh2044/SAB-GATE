import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class DWConv(nn.Module):
    """Depthwise conv 3x3 (padding preserved)."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=True
        )

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.down = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=(kernel_size - 1) // 2,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class Upsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: float = 2.0,
        mode: str = "bilinear",
        align_corners: bool = False,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_up = F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )
        return self.proj(x_up)


class SpectralAdaptiveBasis(nn.Module):
    def __init__(self, in_chans: int, k: int):
        super().__init__()
        self.in_chans = in_chans
        self.k = k
        # per-pixel coefficient conv: C -> k
        self.coef_conv = nn.Conv2d(in_chans, k, kernel_size=1, bias=True)
        # learned spectral basis: shape (C, k)
        self.B_mat = nn.Parameter(torch.randn(in_chans, k) * 0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        N = H * W
        coef_logits = self.coef_conv(x)  # (B, k, H, W)
        coef_logits = coef_logits.view(B, self.k, N)  # (B, k, N)
        Alpha = F.softmax(coef_logits, dim=1)  # (B, k, N)
        # reconstruct: B_mat (C,k) @ Alpha (B,k,N) -> (B,C,N)
        out_flat = torch.einsum("ck,bkn->bcn", self.B_mat, Alpha)
        x_rec = out_flat.view(B, C, H, W)
        return x_rec, Alpha


class GraphTokenPool(nn.Module):
    def __init__(self, in_chans: int, token_count: int, factor_r: int = 16):
        super().__init__()
        self.C = in_chans
        self.m = token_count
        self.r = factor_r
        # V: project C -> r (1x1 conv)
        self.V_proj = nn.Conv2d(in_chans, factor_r, kernel_size=1, bias=True)
        # U: learned m x r matrix
        self.U = nn.Parameter(torch.randn(token_count, factor_r) * 0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        N = H * W
        x_flat = x.view(B, C, N)  # (B, C, N)
        V = self.V_proj(x).view(B, self.r, N)  # (B, r, N)
        L = torch.einsum("mr,brn->bmn", self.U, V)  # (B, m, N)
        S = F.softmax(L, dim=1)  # (B, m, N) soft assignment across tokens
        T = torch.einsum("bcn,bmn->bcm", x_flat, S)  # (B, C, m)
        return T, S


class FactorizedGraphAttention(nn.Module):
    def __init__(
        self,
        in_chans: int,
        embed_d: int = 64,
        d_v: int = 64,
        anchor_r: int = 16,
        token_limit: int = 128,
    ):
        super().__init__()
        self.C = in_chans
        self.d = embed_d
        self.d_v = d_v
        self.r = anchor_r
        self.token_limit = token_limit

        self.Wq = nn.Linear(in_chans, embed_d, bias=False)
        self.Wk = nn.Linear(in_chans, embed_d, bias=False)
        self.Wv = nn.Linear(in_chans, d_v, bias=False)
        self.Wout = nn.Linear(d_v, in_chans, bias=False)

        # anchors R: (r, token_limit) - we'll slice per forward depending on m
        self.R_full = nn.Parameter(torch.randn(self.r, token_limit) * 0.02)

    def forward(self, T: torch.Tensor) -> torch.Tensor:
        B, C, m = T.shape
        assert m <= self.token_limit, "Increase token_limit if needed"
        T_mC = T.transpose(1, 2).contiguous()  # (B, m, C)
        Q = self.Wq(T_mC)  # (B, m, d)
        K = self.Wk(T_mC)  # (B, m, d)
        V = self.Wv(T_mC)  # (B, m, d_v)

        R = self.R_full[:, :m]  # (r, m)
        K_proj = torch.einsum("rm,bjd->brd", R, K)  # (B, r, d)
        V_proj = torch.einsum("rm,bjd->brd", R, V)  # (B, r, d_v)

        scores = torch.einsum("bid,btd->bit", Q, K_proj) / math.sqrt(
            max(1.0, float(self.d))
        )
        A = F.softmax(scores, dim=-1)  # (B, m, r)
        T_out_mdv = torch.einsum("bir,brd->bid", A, V_proj)  # (B, m, d_v)
        T_out_C = self.Wout(T_out_mdv)  # (B, m, C)
        T_out = T_out_C.transpose(1, 2).contiguous()  # (B, C, m)
        return T_out


class TokenToPixelRedistribution(nn.Module):
    """
    Redistribute attended tokens back to pixels via assignment and spectral path; combine with gating.
    """

    def __init__(self, in_chans: int, k: int):
        super().__init__()
        self.C = in_chans
        self.k = k
        self.Pu = nn.Linear(in_chans, k, bias=False)
        self.gate_conv = nn.Conv2d(in_chans, 1, kernel_size=1, bias=True)

    def forward(
        self,
        hatT: torch.Tensor,
        S: torch.Tensor,
        B_mat: torch.Tensor,
        Alpha: torch.Tensor,
        spatial_shape: Tuple[int, int],
    ) -> torch.Tensor:
        Bbatch, C, m = hatT.shape
        H, W = spatial_shape
        N = H * W

        # Path A: assignment-based reconstruction
        X_A_flat = torch.einsum("bcm,bmn->bcn", hatT, S)  # (B, C, N)

        # Path B: spectral-modulated reconstruction
        hatT_mC = hatT.transpose(1, 2).contiguous()  # (B, m, C)
        U_tilde = self.Pu(hatT_mC)  # (B, m, k)
        U_tilde = U_tilde.transpose(1, 2).contiguous()  # (B, k, m)
        US = torch.einsum("bkm,bmn->bkn", U_tilde, S)  # (B, k, N)
        X_B_flat = torch.einsum("ck,bkn->bcn", B_mat, US)  # (B, C, N)

        # Gate per pixel: uses assignment-based spatial feature
        X_A_sp = X_A_flat.view(Bbatch, C, H, W)
        g_logits = self.gate_conv(X_A_sp)  # (B, 1, H, W)
        G_pix = torch.sigmoid(g_logits.view(Bbatch, 1, N))  # (B,1,N)

        X_out_flat = G_pix * X_A_flat + (1.0 - G_pix) * X_B_flat
        X_out = X_out_flat.view(Bbatch, C, H, W)
        return X_out


class DualPathAdapter(nn.Module):
    """
    Local refinement combining depthwise conv path and small MLP path (channel-wise).
    """

    def __init__(self, channels: int, mlp_ratio: int = 2):
        super().__init__()
        self.C = channels
        self.dw = DWConv(channels)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        hidden = max(channels * mlp_ratio, channels + 4)
        self.norm = nn.LayerNorm(channels)
        self.fc1 = nn.Linear(channels, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, channels, bias=True)
        self.fuse = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # depthwise path
        dw = self.dw(x)
        dw = self.pw(dw)
        # MLP path
        x_flat = x.view(B, C, H * W).transpose(1, 2).contiguous()  # (B, N, C)
        y = self.norm(x_flat)
        y = F.gelu(self.fc1(y))
        y = self.fc2(y)  # (B, N, C)
        y = y.transpose(1, 2).contiguous().view(B, C, H, W)
        out = dw + y
        out = self.fuse(out)
        return x + out


class SABGATEBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        k: int,
        tokens: int,
        r: int,
        d: int,
        d_v: int,
        token_limit: int = 128,
    ):
        super().__init__()
        self.sab = SpectralAdaptiveBasis(channels, k)
        self.pool = GraphTokenPool(channels, tokens, factor_r=r)
        self.fga = FactorizedGraphAttention(
            channels, embed_d=d, d_v=d_v, anchor_r=r, token_limit=token_limit
        )
        self.redis = TokenToPixelRedistribution(channels, k)
        self.local = DualPathAdapter(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_rec, Alpha = self.sab(x)  # (B,C,H,W), (B,k,N)
        T, S = self.pool(x_rec)  # (B,C,m), (B,m,N)
        T_hat = self.fga(T)  # (B,C,m)
        x_out = self.redis(T_hat, S, self.sab.B_mat, Alpha, (H, W))
        x_ref = self.local(x_out)
        return x_ref


class SABGATEFormer(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        out_chans: int = 3,
        base_chans: int = 40,
        scales: int = 4,
        k_list: List[int] = None,
        token_list: List[int] = None,
        factor_r: int = 16,
        d: int = 64,
        d_v: int = 64,
    ):
        super().__init__()
        if k_list is None:
            k_list = [12, 14, 16, 18]
        if token_list is None:
            token_list = [64, 128, 128, 128]
        assert len(k_list) == scales and len(token_list) == scales

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.scales = scales

        # initial embed
        self.patch_embed = nn.Conv2d(
            in_chans, base_chans, kernel_size=3, padding=1, bias=True
        )
        # multiscale channels (double each scale)
        self.chs = [base_chans * (2**i) for i in range(scales)]

        # Encoder blocks + downsamplers (PixelUnshuffle then conv)
        enc_blocks = []
        downsamplers = []
        for s in range(scales):
            enc_blocks.append(
                SABGATEBlock(
                    self.chs[s],
                    k=k_list[s],
                    tokens=token_list[s],
                    r=factor_r,
                    d=d,
                    d_v=d_v,
                    token_limit=max(token_list),
                )
            )
            if s < scales - 1:
                downsamplers.append(
                    Downsample(self.chs[s], self.chs[s + 1], kernel_size=3)
                )
        self.encoder = nn.ModuleList(enc_blocks)
        self.downsamplers = nn.ModuleList(downsamplers)

        # Decoder blocks + upsamplers (conv expand then PixelShuffle)
        dec_blocks = []
        upsamplers = []
        # create decoder blocks in reversed scale order so decoder[0] matches bottleneck scale
        for s in reversed(range(scales)):
            dec_blocks.append(
                SABGATEBlock(
                    self.chs[s],
                    k=k_list[s],
                    tokens=token_list[s],
                    r=factor_r,
                    d=d,
                    d_v=d_v,
                    token_limit=max(token_list),
                )
            )
            if s > 0:
                upsamplers.append(
                    Upsample(self.chs[s], self.chs[s - 1], scale_factor=2.0)
                )
        self.decoder = nn.ModuleList(dec_blocks)
        self.upsamplers = nn.ModuleList(upsamplers)

        # refinement at full resolution (a couple of blocks)
        self.refine = nn.Sequential(
            SABGATEBlock(
                self.chs[0],
                k_list[0],
                token_list[0],
                factor_r,
                d,
                d_v,
                token_limit=max(token_list),
            ),
            SABGATEBlock(
                self.chs[0],
                k_list[0],
                token_list[0],
                factor_r,
                d,
                d_v,
                token_limit=max(token_list),
            ),
        )
        self.conv_out = nn.Conv2d(
            self.chs[0], out_chans, kernel_size=3, padding=1, bias=True
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu"
                )
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x0 = self.patch_embed(x)  # (B, chs[0], H, W)

        # Encoder
        feats = []
        cur = x0
        for i, enc in enumerate(self.encoder):
            cur = enc(cur)
            feats.append(cur)
            if i < len(self.downsamplers):
                cur = self.downsamplers[i](cur)

        # Decoder (strict ordering)
        # decoder[0] corresponds to the deepest scale (feats[scales-1])
        for idx in range(self.scales):
            dec_block = self.decoder[idx]
            s = self.scales - 1 - idx
            skip = feats[s]
            if idx == 0:
                # at bottleneck: same resolution as deepest encoder skip
                cur = dec_block(cur + skip)
            else:
                # upsample from coarser level to current skip resolution
                cur = self.upsamplers[idx - 1](cur)
                if cur.shape[-2:] != skip.shape[-2:]:
                    cur = F.interpolate(
                        cur, size=skip.shape[-2:], mode="bilinear", align_corners=False
                    )
                cur = dec_block(cur + skip)

        # refinement + final conv
        out = self.refine(cur + x0)
        out = self.conv_out(out)
        out = x + out
        return torch.clamp(out, 0.0, 1.0)


if __name__ == "__main__":
    device = torch.device("cpu")
    model = SABGATEFormer(in_chans=3, out_chans=3, base_chans=40).to(device)
    inp = torch.randn(2, 3, 128, 128).to(device)
    with torch.no_grad():
        out = model(inp)
    print("Input:", inp.shape, "Output:", out.shape)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
