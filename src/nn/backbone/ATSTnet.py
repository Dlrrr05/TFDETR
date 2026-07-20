"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
ATST backbone adapted from the official Audio-WestlakeU implementation.
"""

import math
import pickle
import warnings
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import get_activation, FrozenBatchNorm2d
from ...core import register


__all__ = ['ATSTBackbone2D']


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_.',
            stacklevel=2,
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        trunc_normal_(self.fc1.weight, std=0.02)
        trunc_normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def get_attention_mask(x, length):
    batch_size, max_len, _ = x.shape
    mask = torch.arange(max_len, device=length.device).expand(batch_size, max_len) >= length[:, None]
    mask = -10000.0 * mask[:, None, None, :]
    return mask.expand(batch_size, 1, max_len, max_len).to(x.device)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        bsz, num_tokens, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, num_tokens, 3, self.num_heads, dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(bsz, num_tokens, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, length=None, return_attention=False):
        mask_att = get_attention_mask(x, length) if length is not None else None
        y, attn = self.attn(self.norm1(x), mask_att)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        return x


class PatchEmbedV2(nn.Module):
    def __init__(self, patch_height=64, patch_width=4, embed_dim=768, input_dim=1):
        super().__init__()
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.input_dim = input_dim
        self.patch_embed = nn.Linear(patch_height * patch_width * input_dim, embed_dim)

    def forward(self, melspec, length=None):
        bsz, channels, height_all, width_all = melspec.shape
        height = height_all - height_all % self.patch_height
        width = width_all - width_all % self.patch_width

        x = melspec[:, :, :height, :width]
        grid_h = height // self.patch_height
        grid_w = width // self.patch_width

        x = x.view(bsz, channels, grid_h, self.patch_height, grid_w, self.patch_width)
        x = x.permute(0, 4, 2, 1, 3, 5).contiguous()
        x = x.view(bsz, grid_w * grid_h, channels * self.patch_height * self.patch_width)
        x = self.patch_embed(x)

        patch_length = None
        if length is not None:
            patch_length = grid_h * ((length - length % self.patch_width) // self.patch_width)

        return x, patch_length, grid_h, grid_w, height, width


class OfficialASTEncoder(nn.Module):
    def __init__(
        self,
        use_cls=True,
        spec_h=64,
        spec_w=1001,
        patch_w=4,
        patch_h=64,
        in_chans=1,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=None,
        pos_type='interp',
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim
        self.spec_w = spec_w
        self.spec_h = spec_h
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.use_cls = use_cls
        self.pos_type = pos_type

        self.patch_embed = PatchEmbedV2(patch_h, patch_w, embed_dim, input_dim=in_chans)
        self.mask_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.num_patches = (spec_h // patch_h) * (spec_w // patch_w)

        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.mask_embed, std=0.02)
        if self.use_cls:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, h, w):
        npatch = x.shape[1] - 1
        num_ref = self.pos_embed.shape[1] - 1
        if npatch == num_ref and w == self.spec_w and h == self.spec_h:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_width
        h0 = h // self.patch_embed.patch_height
        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(
                1,
                self.spec_h // self.patch_h,
                self.spec_w // self.patch_w,
                dim,
            ).permute(0, 3, 1, 2),
            scale_factor=(h0 / (self.spec_h // self.patch_h), w0 / (self.spec_w // self.patch_w)),
            mode='bicubic',
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).contiguous().view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x, length=None):
        bsz, _, h, w = x.shape
        x, patch_length, grid_h, grid_w, crop_h, crop_w = self.patch_embed(x, length)

        if self.use_cls:
            cls_tokens = self.cls_token.expand(bsz, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

            if self.pos_type == 'cut' and x.shape[1] <= self.pos_embed.shape[1]:
                pos = self.pos_embed[:, :x.shape[1], :].expand(bsz, -1, -1)
            else:
                pos = self.interpolate_pos_encoding(x, crop_h, crop_w)
            x = x + pos
        else:
            if self.pos_type == 'cut' and x.shape[1] <= self.pos_embed.shape[1] - 1:
                pos = self.pos_embed[:, 1:x.shape[1] + 1, :].expand(bsz, -1, -1)
            else:
                raise ValueError('ATST backbone expects use_cls=True for interpolated positional encoding.')
            x = x + pos

        x = self.pos_drop(x)
        return x, patch_length, grid_h, grid_w, crop_h, crop_w

    def get_intermediate_layers(self, x, length=None, layer_indices=None):
        if layer_indices is None:
            layer_indices = [len(self.blocks) - 1]

        x, patch_length, grid_h, grid_w, crop_h, crop_w = self.prepare_tokens(x, length=length)

        outputs = []
        layer_set = set(layer_indices)
        for idx, blk in enumerate(self.blocks):
            blk_length = patch_length + 1 if (self.use_cls and patch_length is not None) else patch_length
            x = blk(x, length=blk_length)
            if idx in layer_set:
                outputs.append(self.norm(x))

        return outputs, (grid_h, grid_w), (crop_h, crop_w)


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if padding is None:
            padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)

        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=padding,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class GridProjector(nn.Module):
    def __init__(self, d_model, out_dim=256, stride_f=1, act='relu', refine_depth=2):
        super().__init__()
        blocks = []
        ch_in = d_model
        for i in range(refine_depth):
            stride = (stride_f, 1) if i == 0 else (1, 1)
            blocks.append(
                ConvNormLayer(
                    ch_in,
                    out_dim,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    act=act,
                )
            )
            ch_in = out_dim
        self.refine = nn.Sequential(*blocks)

    def forward(self, x):
        return self.refine(x)


@register()
class ATSTBackbone2D(nn.Module):
    def __init__(
        self,
        pretrained=False,
        arch='auto',
        in_chans=1,
        return_idx=[0, 1, 2],
        hidden_dim=256,
        layers=(3, 7, 11),
        out_strides=(1, 2, 4),
        patch_h=64,
        patch_w=4,
        spec_h=64,
        spec_w=1001,
        embed_dim=None,
        depth=None,
        num_heads=None,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        pos_type='interp',
        resize_height=64,
        min_freq_bins=8,
        act='relu',
        freeze_at=-1,
        freeze_norm=True,
        channel_reduction='mean',
        qkv_bias=False,
    ):
        super().__init__()

        assert len(layers) == 3, 'ATSTBackbone2D expects 3 selected transformer layers.'
        assert len(out_strides) == 3

        self.in_chans = in_chans
        self.return_idx = list(return_idx)
        self.selected_layers_0based = tuple([l - 1 for l in layers])
        self.channel_reduction = channel_reduction
        self.resize_height = resize_height
        self.min_freq_bins = min_freq_bins

        ckpt_raw = None
        if pretrained:
            ckpt_raw = self._load_checkpoint_file(pretrained)

        inferred = self._infer_encoder_config(
            ckpt_raw,
            arch=arch,
            patch_h=patch_h,
            patch_w=patch_w,
            spec_h=spec_h,
            spec_w=spec_w,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )
        self.encoder_cfg = inferred

        self.encoder = OfficialASTEncoder(
            use_cls=True,
            spec_h=inferred['spec_h'],
            spec_w=inferred['spec_w'],
            patch_w=inferred['patch_w'],
            patch_h=inferred['patch_h'],
            in_chans=inferred['in_chans'],
            embed_dim=inferred['embed_dim'],
            depth=inferred['depth'],
            num_heads=inferred['num_heads'],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_type=pos_type,
        )

        self.stage_projs = nn.ModuleList([
            GridProjector(inferred['embed_dim'], out_dim=hidden_dim, stride_f=out_strides[0], act=act, refine_depth=2),
            GridProjector(inferred['embed_dim'], out_dim=hidden_dim, stride_f=out_strides[1], act=act, refine_depth=2),
            GridProjector(inferred['embed_dim'], out_dim=hidden_dim, stride_f=out_strides[2], act=act, refine_depth=2),
        ])

        self.out_channels = [hidden_dim for _ in self.return_idx]
        self.out_strides = [out_strides[i] for i in self.return_idx]

        if pretrained:
            self._load_pretrained(pretrained, ckpt_raw=ckpt_raw)

        if freeze_at >= 0:
            self._freeze_parameters(self.encoder.patch_embed)
            for i in range(min(freeze_at + 1, len(self.encoder.blocks))):
                self._freeze_parameters(self.encoder.blocks[i])

        if freeze_norm:
            self._freeze_norm(self)

    @staticmethod
    def _get_attr(obj, name, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    @staticmethod
    def _load_checkpoint_file(path):
        """
        Official ATST checkpoints are not pure tensor-only state_dict files.
        They may include args / numpy scalar metadata, which PyTorch 2.6+
        rejects when torch.load defaults to weights_only=True.
        """
        try:
            return torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:
            # Older PyTorch versions do not accept the weights_only kwarg.
            return torch.load(path, map_location='cpu')
        except pickle.UnpicklingError as exc:
            raise RuntimeError(
                f'Failed to load ATST checkpoint: {path}. '
                'This checkpoint likely requires torch.load(..., weights_only=False).'
            ) from exc

    def _infer_encoder_config(
        self,
        ckpt_raw,
        arch='auto',
        patch_h=64,
        patch_w=4,
        spec_h=64,
        spec_w=1001,
        embed_dim=None,
        depth=None,
        num_heads=None,
    ):
        args = self._get_attr(ckpt_raw, 'args', None) if isinstance(ckpt_raw, dict) else None

        if arch == 'small':
            embed_dim = embed_dim or 384
            depth = depth or 12
            num_heads = num_heads or 6
        elif arch == 'base':
            embed_dim = embed_dim or 768
            depth = depth or 12
            num_heads = num_heads or 12

        if args is not None:
            patch_h = self._get_attr(args, 'patch_h', patch_h)
            patch_w = self._get_attr(args, 'patch_w', patch_w)
            spec_h = self._get_attr(args, 'n_mels', spec_h)

        state_dict = self._extract_encoder_state_dict(ckpt_raw, model_keys=None) if ckpt_raw is not None else None
        if state_dict:
            if embed_dim is None:
                if 'pos_embed' in state_dict:
                    embed_dim = state_dict['pos_embed'].shape[-1]
                elif 'patch_embed.patch_embed.weight' in state_dict:
                    embed_dim = state_dict['patch_embed.patch_embed.weight'].shape[0]

            if depth is None:
                block_ids = []
                for key in state_dict.keys():
                    if key.startswith('blocks.'):
                        try:
                            block_ids.append(int(key.split('.')[1]))
                        except (IndexError, ValueError):
                            pass
                if block_ids:
                    depth = max(block_ids) + 1

            if num_heads is None and embed_dim is not None:
                default_heads = {384: 6, 768: 12, 1024: 16}
                num_heads = default_heads.get(embed_dim, 8)

            if 'pos_embed' in state_dict and spec_w == 1001:
                num_patch_tokens = state_dict['pos_embed'].shape[1] - 1
                grid_h = max(spec_h // patch_h, 1)
                if grid_h > 0 and num_patch_tokens % grid_h == 0:
                    grid_w = num_patch_tokens // grid_h
                    spec_w = grid_w * patch_w

        embed_dim = embed_dim or 384
        depth = depth or 12
        num_heads = num_heads or 6

        return {
            'embed_dim': embed_dim,
            'depth': depth,
            'num_heads': num_heads,
            'patch_h': patch_h,
            'patch_w': patch_w,
            'spec_h': spec_h,
            'spec_w': spec_w,
            'in_chans': self.in_chans,
        }

    @staticmethod
    def _is_state_dict_like(d):
        return isinstance(d, dict) and len(d) > 0 and any(torch.is_tensor(v) for v in d.values())

    def _extract_encoder_state_dict(self, ckpt_raw, model_keys=None):
        if ckpt_raw is None:
            return None

        candidates = []

        def collect(mapping, depth=0):
            if not isinstance(mapping, dict) or depth > 2:
                return
            if self._is_state_dict_like(mapping):
                candidates.append(mapping)
            for value in mapping.values():
                if isinstance(value, dict):
                    collect(value, depth + 1)

        collect(ckpt_raw)

        prefixes = [
            'model.teacher.encoder.',
            'teacher.encoder.',
            'model.student.encoder.',
            'student.encoder.',
            'encoder.encoder.',
            'module.teacher.encoder.',
            'module.student.encoder.',
            'module.encoder.',
            'backbone.',
            'module.',
            'teacher.',
            'student.',
            'encoder.',
            '',
        ]
        core_names = ('patch_embed.', 'cls_token', 'pos_embed', 'mask_embed', 'blocks.', 'norm.')

        best_clean = None
        best_score = -1

        for cand in candidates:
            for prefix in prefixes:
                clean = {}
                for key, value in cand.items():
                    if not torch.is_tensor(value):
                        continue
                    new_key = key
                    if prefix and new_key.startswith(prefix):
                        new_key = new_key[len(prefix):]
                    elif prefix:
                        continue
                    if new_key.startswith(core_names):
                        clean[new_key] = value

                if not clean:
                    continue

                if model_keys is not None:
                    score = sum(1 for key in clean.keys() if key in model_keys)
                else:
                    score = len(clean)

                if score > best_score:
                    best_score = score
                    best_clean = clean

        return best_clean

    def _load_pretrained(self, pretrained, ckpt_raw=None):
        if ckpt_raw is None:
            ckpt_raw = self._load_checkpoint_file(pretrained)

        model_state = self.encoder.state_dict()
        state_dict = self._extract_encoder_state_dict(ckpt_raw, model_keys=set(model_state.keys()))
        if not state_dict:
            raise ValueError(f'Failed to extract ATST encoder state_dict from checkpoint: {pretrained}')

        loadable = {}
        skipped = []
        for key, value in state_dict.items():
            if key not in model_state:
                continue
            if model_state[key].shape != value.shape:
                skipped.append(f'{key}: ckpt{tuple(value.shape)} != model{tuple(model_state[key].shape)}')
                continue
            loadable[key] = value

        msg = self.encoder.load_state_dict(loadable, strict=False)
        print(f'Load ATSTBackbone2D encoder from {pretrained}')
        print(f'  missing={len(msg.missing_keys)} | unexpected={len(msg.unexpected_keys)} | loaded={len(loadable)}')
        if skipped:
            print(f'  skipped_shape_mismatch={len(skipped)}')
            for item in skipped[:5]:
                print(f'    {item}')

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                new_child = self._freeze_norm(child)
                if new_child is not child:
                    setattr(m, name, new_child)
        return m

    def _prepare_spec(self, x):
        if isinstance(x, (list, tuple)):
            x = x[0]

        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() != 4:
            raise ValueError(f'Unsupported ATST input shape: {x.shape}')

        if x.shape[1] == self.in_chans:
            spec = x
        elif self.in_chans == 1:
            if self.channel_reduction == 'mean':
                spec = x.mean(dim=1, keepdim=True)
            elif self.channel_reduction == 'first':
                spec = x[:, :1]
            else:
                raise ValueError(f'Unsupported channel_reduction: {self.channel_reduction}')
        else:
            raise ValueError(f'Cannot adapt input shape {x.shape} to in_chans={self.in_chans}')

        if self.resize_height is not None and spec.shape[-2] != self.resize_height:
            spec = F.interpolate(
                spec,
                size=(self.resize_height, spec.shape[-1]),
                mode='bilinear',
                align_corners=False,
            )

        return spec

    @staticmethod
    def _tokens_to_grid(tokens, grid_h, grid_w, use_cls=True):
        if use_cls:
            tokens = tokens[:, 1:, :]
        bsz, num_tokens, dim = tokens.shape
        if num_tokens != grid_h * grid_w:
            raise RuntimeError(f'Cannot reshape {num_tokens} tokens into grid ({grid_h}, {grid_w}).')
        tokens = tokens.view(bsz, grid_w, grid_h, dim).permute(0, 2, 1, 3).contiguous()
        return tokens

    def _lift_frequency_if_needed(self, grid):
        bsz, grid_h, grid_w, dim = grid.shape
        if self.min_freq_bins is None or grid_h >= self.min_freq_bins:
            return grid
        x = grid.permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, size=(self.min_freq_bins, grid_w), mode='bilinear', align_corners=False)
        return x.permute(0, 2, 3, 1).contiguous()

    def forward(self, x):
        spec = self._prepare_spec(x)
        hidden_states, (grid_h, grid_w), _ = self.encoder.get_intermediate_layers(
            spec,
            length=None,
            layer_indices=self.selected_layers_0based,
        )

        outs = []
        for idx, (hidden, proj) in enumerate(zip(hidden_states, self.stage_projs)):
            grid = self._tokens_to_grid(hidden, grid_h=grid_h, grid_w=grid_w, use_cls=self.encoder.use_cls)
            grid = self._lift_frequency_if_needed(grid)
            feat = grid.permute(0, 3, 1, 2).contiguous()
            feat = proj(feat)
            if idx in self.return_idx:
                outs.append(feat)

        return outs
