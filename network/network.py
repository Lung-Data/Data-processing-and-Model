from __future__ import division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from sam2.build_sam import build_sam2
import numpy as np
from typing import Union, Sequence, Tuple, Optional, Any, Callable


def pair(Val):
    return Val if isinstance(Val, (tuple, list)) else (Val, Val)


def makeDivisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act:
            x = self.relu(x)
        return x


class DecoupleLayer(nn.Module):
    def __init__(self, in_c=1024, out_c=256):
        super().__init__()
        mid_c = min(512, in_c * 2)
        self.cbr_fg = nn.Sequential(
            CBR(in_c, mid_c, kernel_size=3, padding=1),
            CBR(mid_c, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        self.cbr_bg = nn.Sequential(
            CBR(in_c, mid_c, kernel_size=3, padding=1),
            CBR(mid_c, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )

    def forward(self, x):
        return self.cbr_fg(x), self.cbr_bg(x)


class FBCA_Block(nn.Module):
    def __init__(self, in_c, out_c=128, num_heads=4, kernel_size=3, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = out_c

        self.input_conv = nn.Sequential(
            CBR(in_c, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=3, padding=1),
        )

        self.decouple = DecoupleLayer(in_c, out_c)

        self.fg_attention = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=kernel_size, padding=kernel_size // 2, groups=out_c),
            nn.Sigmoid()
        )

        self.bg_attention = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=kernel_size, padding=kernel_size // 2, groups=out_c),
            nn.Sigmoid()
        )

        self.feature_fusion = nn.Sequential(
            nn.Conv2d(out_c * 2, out_c, kernel_size=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

        self.output_conv = nn.Sequential(
            CBR(out_c, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=3, padding=1),
        )

        self.dropout = nn.Dropout2d(proj_drop)

    def forward(self, x, fg=None, bg=None):
        x_processed = self.input_conv(x)

        if fg is None or bg is None:
            fg, bg = self.decouple(x)

        B, C, H, W = x_processed.shape
        if fg.shape[2:] != (H, W):
            fg = F.interpolate(fg, size=(H, W), mode='bilinear', align_corners=False)
        if bg.shape[2:] != (H, W):
            bg = F.interpolate(bg, size=(H, W), mode='bilinear', align_corners=False)

        if fg.shape[1] != self.dim:
            fg = F.adaptive_avg_pool2d(fg, 1).expand(-1, self.dim, H, W)
        if bg.shape[1] != self.dim:
            bg = F.adaptive_avg_pool2d(bg, 1).expand(-1, self.dim, H, W)

        fg_enhanced = x_processed * self.fg_attention(fg)
        bg_enhanced = x_processed * (1.0 - self.bg_attention(bg))

        fused = self.feature_fusion(torch.cat([fg_enhanced, bg_enhanced], dim=1))
        output = self.output_conv(x_processed + fused)

        return self.dropout(output)


class BaseConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: Optional[int] = None, groups: int = 1, bias: Optional[bool] = None,
                 BNorm: bool = False, ActLayer: Optional[Callable[..., nn.Module]] = None,
                 dilation: int = 1, Momentum: float = 0.1, **kwargs: Any) -> None:
        super().__init__()
        if padding is None:
            padding = int((kernel_size - 1) // 2 * dilation)
        if bias is None:
            bias = not BNorm

        self.Conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, **kwargs)
        self.Bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=Momentum) if BNorm else nn.Identity()

        if ActLayer is not None:
            self.Act = ActLayer() if isinstance(list(ActLayer().named_modules())[0][1], nn.Sigmoid) else ActLayer(
                inplace=True)
        else:
            self.Act = None

    def forward(self, x):
        x = self.Conv(x)
        x = self.Bn(x)
        if self.Act is not None:
            x = self.Act(x)
        return x


class LinearSelfAttention(nn.Module):
    def __init__(self, DimEmbed: int, AttnDropRate: float = 0.0, Bias: bool = True) -> None:
        super().__init__()
        self.qkv_proj = BaseConv2d(DimEmbed, 1 + (2 * DimEmbed), 1, bias=Bias)
        self.AttnDropRate = nn.Dropout(p=AttnDropRate)
        self.out_proj = BaseConv2d(DimEmbed, DimEmbed, 1, bias=Bias)
        self.DimEmbed = DimEmbed

    def forward(self, x):
        qkv = self.qkv_proj(x)
        query, key, value = torch.split(qkv, [1, self.DimEmbed, self.DimEmbed], dim=1)
        context_scores = self.AttnDropRate(F.softmax(query, dim=-1))
        context_vector = torch.sum(key * context_scores, dim=-1, keepdim=True)
        return self.out_proj(F.relu(value) * context_vector.expand_as(value))


class LinearAttnFFN(nn.Module):
    def __init__(self, DimEmbed: int, DimFfnLatent: int, AttnDropRate: float = 0.0,
                 DropRate: float = 0.1, FfnDropRate: float = 0.0) -> None:
        super().__init__()
        self.PreNormAttn = nn.Sequential(
            nn.BatchNorm2d(DimEmbed),
            LinearSelfAttention(DimEmbed, AttnDropRate, Bias=True),
            nn.Dropout(DropRate),
        )
        self.PreNormFfn = nn.Sequential(
            nn.BatchNorm2d(DimEmbed),
            BaseConv2d(DimEmbed, DimFfnLatent, 1, 1, ActLayer=nn.SiLU),
            nn.Dropout(FfnDropRate),
            BaseConv2d(DimFfnLatent, DimEmbed, 1, 1),
            nn.Dropout(DropRate),
        )

    def forward(self, x):
        return x + self.PreNormFfn(x + self.PreNormAttn(x))


class PatchBlock(nn.Module):
    def __init__(self, InChannels: int, FfnMultiplier: Union[Sequence, int, float] = 2.0,
                 NumAttnBlocks: int = 1, AttnDropRate: float = 0.0, DropRate: float = 0.0,
                 FfnDropRate: float = 0.0, PatchRes: int = 2, Dilation: int = 1, SDProb: float = 0.0) -> None:
        super().__init__()

        DimCNNOut = InChannels // 2
        self.LocalRep = nn.Sequential(
            BaseConv2d(InChannels, InChannels, 3, 1, dilation=Dilation, BNorm=True, ActLayer=nn.SiLU),
            BaseConv2d(InChannels, DimCNNOut, 1, 1, bias=False)
        )

        self.GlobalRep, _ = self.buildAttnLayer(DimCNNOut, FfnMultiplier, NumAttnBlocks, AttnDropRate, DropRate,
                                                FfnDropRate)
        self.ConvProj = BaseConv2d(2 * DimCNNOut, InChannels, 1, 1, BNorm=True)

        self.HPatch, self.WPatch = pair(PatchRes)
        self.Dropout = DropPath(SDProb) if SDProb > 0. else nn.Identity()

    def buildAttnLayer(self, DimModel: int, FfnMult: Union[Sequence, int, float], NumAttnBlocks: int,
                       AttnDropRate: float, DropRate: float, FfnDropRate: float) -> Tuple[nn.Module, int]:
        if isinstance(FfnMult, Sequence) and len(FfnMult) == 2:
            DimFfn = np.linspace(FfnMult[0], FfnMult[1], NumAttnBlocks, dtype=float) * DimModel
        elif isinstance(FfnMult, Sequence) and len(FfnMult) == 1:
            DimFfn = [FfnMult[0] * DimModel] * NumAttnBlocks
        else:
            DimFfn = [FfnMult * DimModel] * NumAttnBlocks

        DimFfn = [makeDivisible(d, 16) for d in DimFfn]
        GlobalRep = [LinearAttnFFN(DimModel, int(DimFfn[i]), AttnDropRate, DropRate, FfnDropRate) for i in
                     range(NumAttnBlocks)]
        GlobalRep.append(nn.BatchNorm2d(DimModel))
        return nn.Sequential(*GlobalRep), DimModel

    def forward(self, x):
        FmConv = self.LocalRep(x)
        B, C, H, W = FmConv.shape
        Patches = F.unfold(FmConv, kernel_size=(self.HPatch, self.WPatch), stride=(self.HPatch, self.WPatch))
        Patches = Patches.reshape(B, C, self.HPatch * self.WPatch, -1)
        Patches = self.GlobalRep(Patches)
        Patches = Patches.reshape(B, C * self.HPatch * self.WPatch, -1)
        Fm = F.fold(Patches, output_size=(H, W), kernel_size=(self.HPatch, self.WPatch),
                    stride=(self.HPatch, self.WPatch))
        return x + self.Dropout(self.ConvProj(torch.cat((Fm, FmConv), dim=1)))


class InputChannelAdapter(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        if input_channels == 1:
            self.channel_adapter = nn.Conv2d(1, 3, kernel_size=1, bias=False)
            nn.init.constant_(self.channel_adapter.weight, 1.0 / 3.0)
        else:
            self.channel_adapter = nn.Identity()

    def forward(self, x):
        return self.channel_adapter(x)


class Adapter(nn.Module):
    def __init__(self, blk, bottleneck_dim=128, dropout=0.1):
        super().__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features

        self.adapter = nn.Sequential(
            nn.Linear(dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, bottleneck_dim // 2),
            nn.LayerNorm(bottleneck_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim // 2, dim),
        )

        self.scale = nn.Parameter(torch.ones(1) * 0.1)

        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        nn.init.xavier_uniform_(self.adapter[0].weight)
        nn.init.xavier_uniform_(self.adapter[4].weight)

    def forward(self, x):
        return self.block(x + self.scale * self.adapter(x))


class DimensionAdapter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.adapter(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class GlobalSparseAttn(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr = sr_ratio
        if self.sr > 1:
            self.sampler = nn.AvgPool2d(1, sr_ratio)
            self.LocalProp = nn.ConvTranspose2d(dim, dim, sr_ratio, stride=sr_ratio, groups=dim)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sampler = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x, H: int, W: int):
        B, N, C = x.shape
        if self.sr > 1:
            x = self.sampler(x.transpose(1, 2).reshape(B, C, H, W)).flatten(2).transpose(1, 2)

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = self.attn_drop((q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)

        if self.sr > 1:
            x = self.norm(
                self.LocalProp(x.permute(0, 2, 1).reshape(B, C, int(H / self.sr), int(W / self.sr))).reshape(B, C,
                                                                                                             -1).permute(
                    0, 2, 1))

        return self.proj_drop(self.proj(x))


class LocalAgg(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = CMlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class SelfAttn(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1.):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = GlobalSparseAttn(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, N, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x.transpose(1, 2).reshape(B, N, H, W)


class LGF_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1., use_patchblock=True):
        super().__init__()

        self.LocalAgg = LocalAgg(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path,
                                 act_layer, norm_layer) if sr_ratio > 1 else nn.Identity()
        self.SelfAttn = SelfAttn(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path,
                                 act_layer, norm_layer, sr_ratio)

        if use_patchblock:
            if dim <= 48:
                patch_res, num_blocks, ffn_mult = 4, 1, 1.5
            elif dim <= 96:
                patch_res, num_blocks, ffn_mult = 2, 1, 2.0
            else:
                patch_res, num_blocks, ffn_mult = 2, 1, 2.5

            self.PatchBlock = PatchBlock(InChannels=dim, FfnMultiplier=ffn_mult, NumAttnBlocks=num_blocks,
                                         AttnDropRate=attn_drop, DropRate=drop, FfnDropRate=drop,
                                         PatchRes=patch_res, SDProb=drop_path * 0.5)
        else:
            self.PatchBlock = nn.Identity()

    def forward(self, x):
        return self.PatchBlock(self.SelfAttn(self.LocalAgg(x)))


class Encoder(nn.Module):
    def __init__(self, input_channels=1, checkpoint_path=None, freeze_encoder=True):
        super().__init__()

        self.input_adapter = InputChannelAdapter(input_channels)

        model = build_sam2("sam2_hiera_l.yaml", checkpoint_path)
        del model.sam_mask_decoder, model.sam_prompt_encoder, model.memory_encoder
        del model.memory_attention, model.mask_downsample, model.obj_ptr_tpos_proj
        del model.obj_ptr_proj, model.image_encoder.neck

        self.hiera_encoder = model.image_encoder.trunk

        if freeze_encoder:
            for param in self.hiera_encoder.parameters():
                param.requires_grad = False

        enhanced_blocks = [
            Adapter(block, bottleneck_dim=96 if i < len(self.hiera_encoder.blocks) // 2 else 128, dropout=0.1)
            for i, block in enumerate(self.hiera_encoder.blocks)]
        self.hiera_encoder.blocks = nn.Sequential(*enhanced_blocks)

        self.init_conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Dropout(0.02),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU()
        )

        self.dim_adapter1 = DimensionAdapter(144, 48)
        self.dim_adapter2 = DimensionAdapter(288, 96)
        self.dim_adapter3 = DimensionAdapter(576, 192)
        self.dim_adapter4 = DimensionAdapter(1152, 384)

        self.lgf_en1 = LGF_Block(dim=48, num_heads=3, mlp_ratio=4., qkv_bias=True, drop=0.05, sr_ratio=4,
                                 drop_path=0.05)
        self.lgf_en2 = LGF_Block(dim=96, num_heads=6, mlp_ratio=4., qkv_bias=True, drop=0.1, sr_ratio=2, drop_path=0.05)
        self.lgf_en3 = LGF_Block(dim=192, num_heads=12, mlp_ratio=4., qkv_bias=True, drop=0.15, sr_ratio=2,
                                 drop_path=0.05)
        self.lgf_en4 = LGF_Block(dim=384, num_heads=24, mlp_ratio=4., qkv_bias=True, drop=0.2, sr_ratio=1,
                                 drop_path=0.05)

    def forward(self, x):
        x0 = self.init_conv(x)
        sam2_features = self.hiera_encoder(self.input_adapter(x))

        x1 = F.interpolate(self.dim_adapter1(sam2_features[0]), size=(x.shape[2] // 2, x.shape[3] // 2),
                           mode='bilinear', align_corners=True)
        x2 = F.interpolate(self.dim_adapter2(sam2_features[1]), size=(x.shape[2] // 4, x.shape[3] // 4),
                           mode='bilinear', align_corners=True)
        x3 = F.interpolate(self.dim_adapter3(sam2_features[2]), size=(x.shape[2] // 8, x.shape[3] // 8),
                           mode='bilinear', align_corners=True)
        x4 = F.interpolate(self.dim_adapter4(sam2_features[3]), size=(x.shape[2] // 16, x.shape[3] // 16),
                           mode='bilinear', align_corners=True)

        return [x0, self.lgf_en1(x1), self.lgf_en2(x2), self.lgf_en3(x3), self.lgf_en4(x4)]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p):
        super().__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        return self.conv(torch.cat([x2, self.up(x1)], dim=1))


class Decoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.ft_chns = params['feature_chns']

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.fbca4 = FBCA_Block(in_c=self.ft_chns[4], out_c=self.ft_chns[4], num_heads=8, kernel_size=3, attn_drop=0.15,
                                proj_drop=0.15)
        self.fbca3 = FBCA_Block(in_c=self.ft_chns[3], out_c=self.ft_chns[3], num_heads=6, kernel_size=3, attn_drop=0.15,
                                proj_drop=0.12)
        self.fbca2 = FBCA_Block(in_c=self.ft_chns[2], out_c=self.ft_chns[2], num_heads=4, kernel_size=5, attn_drop=0.1,
                                proj_drop=0.1)
        self.fbca1 = FBCA_Block(in_c=self.ft_chns[1], out_c=self.ft_chns[1], num_heads=3, kernel_size=5, attn_drop=0.05,
                                proj_drop=0.05)

    def forward(self, feature):
        x0, x1, x2, x3, x4 = feature

        x = self.up1(self.fbca4(x4), self.fbca3(x3))
        p = [x]
        x = self.up2(x, self.fbca2(x2))
        p.append(x)
        x = self.up3(x, self.fbca1(x1))
        p.append(x)
        x = self.up4(x, x0)
        p.append(x)

        return p


class CIE_Head(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.ft_chns = params['feature_chns']
        self.n_class = params['class_num']

        self.out_head1 = nn.Conv2d(self.ft_chns[3], self.n_class, 1)
        self.out_head2 = nn.Conv2d(self.ft_chns[2], self.n_class, 1)
        self.out_head3 = nn.Conv2d(self.ft_chns[1], self.n_class, 1)
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        return [
            F.interpolate(self.out_head1(feature[0]), scale_factor=8, mode='bilinear', align_corners=True),
            F.interpolate(self.out_head2(feature[1]), scale_factor=4, mode='bilinear', align_corners=True),
            F.interpolate(self.out_head3(feature[2]), scale_factor=2, mode='bilinear', align_corners=True),
            self.out_conv(feature[3])
        ]


class NetworkCC(nn.Module):
    def __init__(self, in_chns, class_num, sam2_checkpoint_path=None, freeze_sam2=True):
        super().__init__()

        params = {
            'in_chns': in_chns,
            'feature_chns': [16, 48, 96, 192, 384],
            'class_num': class_num,
            'bilinear': False,
        }

        self.encoder = Encoder(input_channels=in_chns, checkpoint_path=sam2_checkpoint_path, freeze_encoder=freeze_sam2)
        self.decoder = Decoder(params)
        self.cie = CIE_Head(params)

    def forward(self, x):
        return self.cie(self.decoder(self.encoder(x)))


if __name__ == "__main__":
    model = NetworkCC(in_chns=1, class_num=2, sam2_checkpoint_path=None, freeze_sam2=True)
    x = torch.randn(2, 1, 352, 352)
    outputs = model(x)
    print(f"Model output shapes: {[out.shape for out in outputs]}")