"""
SwinPAR: Swin Transformer + MobileNetV2 Hybrid Model for Person Attribute Recognition
Model Architecture and Core Components

This file contains all the model architecture, dataset classes, and utility functions
needed for the SwinPAR person attribute recognition system.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import cv2
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from sklearn import metrics
import math
import os

# Person Attribute Labels (40 attributes)
LABEL_COLUMNS = [
    'Age-Young', 'Age-Adult', 'Age-Old', 'Gender-Female',
    'Hair-Length-Short', 'Hair-Length-Long', 'Hair-Length-Bald',
    'UpperBody-Length-Short', 'UpperBody-Color-Black',
    'UpperBody-Color-Blue', 'UpperBody-Color-Brown',
    'UpperBody-Color-Green', 'UpperBody-Color-Grey',
    'UpperBody-Color-Orange', 'UpperBody-Color-Pink',
    'UpperBody-Color-Purple', 'UpperBody-Color-Red',
    'UpperBody-Color-White', 'UpperBody-Color-Yellow',
    'UpperBody-Color-Other', 'LowerBody-Length-Short',
    'LowerBody-Color-Black', 'LowerBody-Color-Blue',
    'LowerBody-Color-Brown', 'LowerBody-Color-Green',
    'LowerBody-Color-Grey', 'LowerBody-Color-Orange',
    'LowerBody-Color-Pink', 'LowerBody-Color-Purple', 'LowerBody-Color-Red',
    'LowerBody-Color-White', 'LowerBody-Color-Yellow',
    'LowerBody-Color-Other', 'LowerBody-Type-Trousers&Shorts',
    'LowerBody-Type-Skirt&Dress', 'Accessory-Backpack', 'Accessory-Bag',
    'Accessory-Glasses-Normal', 'Accessory-Glasses-Sun', 'Accessory-Hat'
]

# =============================================================================
# SwinV2 Transformer Components
# =============================================================================

class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    """Partition into non-overlapping windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reverse of window partition."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block."""

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Identity() if drop_path <= 0. else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer."""

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage."""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                               num_heads=num_heads, window_size=window_size,
                               shift_size=0 if (i % 2 == 0) else window_size // 2,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                               norm_layer=norm_layer)
            for i in range(depth)])

        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


# =============================================================================
# Attention Mechanisms (SE, CBAM)
# =============================================================================

class LightweightSEBlock(nn.Module):
    """Lightweight Squeeze-and-Excitation block"""
    def __init__(self, channel, reduction=8):
        super(LightweightSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(channel // reduction, 8), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channel // reduction, 8), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class LightweightChannelAttention(nn.Module):
    """Lightweight Channel Attention"""
    def __init__(self, in_planes, ratio=8):
        super(LightweightChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, max(in_planes // ratio, 8), 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(max(in_planes // ratio, 8), in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class LightweightSpatialAttention(nn.Module):
    """Lightweight Spatial Attention"""
    def __init__(self, kernel_size=3):
        super(LightweightSpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 1 if kernel_size == 3 else 3
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_cat = self.conv1(x_cat)
        return self.sigmoid(x_cat)


class LightweightCBAM(nn.Module):
    """Lightweight Convolutional Block Attention Module"""
    def __init__(self, in_planes, ratio=8, kernel_size=3):
        super(LightweightCBAM, self).__init__()
        self.ca = LightweightChannelAttention(in_planes, ratio)
        self.sa = LightweightSpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# =============================================================================
# MobileNetV2 Components
# =============================================================================

class LightweightInvertedResidual(nn.Module):
    """Lightweight Inverted Residual Block with SE"""
    def __init__(self, inp, oup, stride, expand_ratio, use_se=True):
        super(LightweightInvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        layers.extend([
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])

        self.conv = nn.Sequential(*layers)
        self.se = LightweightSEBlock(oup, reduction=8) if use_se else nn.Identity()

    def forward(self, x):
        if self.use_res_connect:
            out = self.conv(x)
            out = self.se(out)
            return x + out
        else:
            out = self.conv(x)
            out = self.se(out)
            return out


# =============================================================================
# Main SwinPAR Model
# =============================================================================

class SwinPAR(nn.Module):
    """SwinPAR: Lightweight Hybrid SwinV2 + MobileNetV2 for Person Attribute Recognition"""
    
    def __init__(self, img_size=224, num_classes=40, use_pretrained_swin=True):
        super(SwinPAR, self).__init__()

        # Minimal SwinV2 backbone
        self.swin_backbone = self._create_minimal_swin_backbone(img_size, use_pretrained_swin)

        # Lightweight transition layer
        self.transition = nn.Sequential(
            nn.Conv2d(48, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU6(inplace=True)
        )

        # Lightweight MobileNetV2 with SE blocks
        self.mobilenet_layers = self._create_lightweight_mobilenet()

        # Lightweight CBAM attention
        self.cbam = LightweightCBAM(160)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(160, num_classes),
            nn.Sigmoid()
        )

    def _create_minimal_swin_backbone(self, img_size, use_pretrained):
        """Create minimal SwinV2 backbone"""
        embed_dim = 48

        # Patch embedding
        patch_embed = PatchEmbed(
            img_size=img_size, patch_size=4, in_chans=3, embed_dim=embed_dim,
            norm_layer=nn.LayerNorm
        )

        # Position embedding
        num_patches = patch_embed.num_patches
        absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        pos_drop = nn.Dropout(p=0.0)

        # Lightweight transformer block
        patches_resolution = patch_embed.patches_resolution
        stage1 = BasicLayer(
            dim=embed_dim,
            input_resolution=patches_resolution,
            depth=1,
            num_heads=2,
            window_size=7,
            mlp_ratio=2.,
            qkv_bias=True,
            drop=0.,
            attn_drop=0.,
            drop_path=0.05,
            norm_layer=nn.LayerNorm,
            downsample=None
        )

        # Final norm
        norm = nn.LayerNorm(embed_dim)

        # Combine into sequential model
        backbone = nn.ModuleDict({
            'patch_embed': patch_embed,
            'pos_drop': pos_drop,
            'stage1': stage1,
            'norm': norm
        })
        backbone.absolute_pos_embed = absolute_pos_embed

        # Load pretrained weights if available
        if use_pretrained:
            self._load_minimal_pretrained_weights(backbone)

        return backbone

    def _load_minimal_pretrained_weights(self, backbone):
        """Load compatible pretrained SwinV2 weights"""
        try:
            import timm
            print("Loading minimal pretrained SwinV2 weights...")
            pretrained_model = timm.create_model('swinv2_tiny_window16_256', pretrained=True, num_classes=1000)

            backbone_dict = backbone.state_dict()
            pretrained_dict = pretrained_model.state_dict()

            compatible_weights = {}
            for k, v in pretrained_dict.items():
                if k.startswith('patch_embed'):
                    if k == 'patch_embed.proj.weight' and v.shape[0] == 96:
                        compatible_weights[k] = v[:48, :, :, :]
                    elif k == 'patch_embed.norm.weight' and v.shape[0] == 96:
                        compatible_weights[k] = v[:48]
                    elif k == 'patch_embed.norm.bias' and v.shape[0] == 96:
                        compatible_weights[k] = v[:48]
                    elif k in backbone_dict and backbone_dict[k].shape == v.shape:
                        compatible_weights[k] = v
                elif k == 'pos_embed' and v.shape[-1] == 96:
                    compatible_weights['absolute_pos_embed'] = v[:, :, :48]

            if compatible_weights:
                backbone_dict.update(compatible_weights)
                backbone.load_state_dict(backbone_dict)
                print(f"Loaded {len(compatible_weights)} compatible pretrained parameters")
            else:
                print("No compatible pretrained weights found, using random initialization")

        except Exception as e:
            print(f"Could not load pretrained SwinV2 weights: {e}")
            print("Initializing minimal SwinV2 backbone with random weights...")

    def _create_lightweight_mobilenet(self):
        """Create lightweight MobileNetV2 layers"""
        configs = [
            [2, 16, 1, 1],   # 24 -> 16
            [3, 24, 1, 2],   # 16 -> 24, downsample
            [3, 32, 2, 2],   # 24 -> 32, downsample
            [4, 48, 2, 2],   # 32 -> 48, downsample
            [4, 80, 1, 1],   # 48 -> 80
            [6, 160, 1, 1],  # 80 -> 160
        ]

        layers = []
        input_channel = 24

        for t, c, n, s in configs:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(LightweightInvertedResidual(input_channel, c, stride, t, use_se=True))
                input_channel = c

        return nn.Sequential(*layers)

    def forward(self, x):
        # SwinV2 backbone
        x = self.swin_backbone['patch_embed'](x)
        x = x + self.swin_backbone.absolute_pos_embed
        x = self.swin_backbone['pos_drop'](x)
        x = self.swin_backbone['stage1'](x)
        x = self.swin_backbone['norm'](x)

        # Reshape for conv layers
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)

        # Transition to MobileNetV2
        x = self.transition(x)

        # MobileNetV2 with SE blocks
        x = self.mobilenet_layers(x)

        # CBAM attention
        x = self.cbam(x)

        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Classification
        x = self.classifier(x)

        return x


# =============================================================================
# Dataset and Transforms
# =============================================================================

class SwinPARDataset(Dataset):
    """Dataset class for SwinPAR model"""
    
    def __init__(self, image_paths, targets, transform=None):
        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        if image is None:
            raise ValueError(f"Failed to load image: {self.image_paths[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target = self.targets[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return {
            'image': image,
            'targets': torch.tensor(target, dtype=torch.float32)
        }


def get_swinpar_transforms():
    """Get transforms for SwinPAR model"""
    train_transform = A.Compose([
        A.Resize(width=224, height=224, always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=10, p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    valid_transform = A.Compose([
        A.Resize(width=224, height=224, always_apply=True),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    return train_transform, valid_transform


# =============================================================================
# Model Creation and Utility Functions
# =============================================================================

def create_swinpar_model(num_classes=40, img_size=224, pretrained=True):
    """Create SwinPAR model"""
    model = SwinPAR(
        img_size=img_size,
        num_classes=num_classes,
        use_pretrained_swin=pretrained
    )

    param_count = count_parameters(model)
    print(f"Created SwinPAR model:")
    print(f"  - Total parameters: {param_count:,}")
    print(f"  - Image size: {img_size}x{img_size}")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Pretrained SwinV2: {pretrained}")

    return model


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model):
    """Get detailed model summary"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate model size in MB
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    model_size = (param_size + buffer_size) / 1024 / 1024

    summary = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params,
        'model_size_mb': model_size
    }

    print("Model Summary:")
    print("-" * 50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Model size: {model_size:.2f} MB")

    return summary


def save_model(model, path, additional_info=None):
    """Save model with additional information"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_name': 'SwinPAR',
        'label_columns': LABEL_COLUMNS,
        'num_classes': len(LABEL_COLUMNS),
        'img_size': 224,
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, path)
    print(f"Model saved to: {path}")


def load_model(path, device='cpu'):
    """Load SwinPAR model from checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    
    # Create model
    model = create_swinpar_model(
        num_classes=checkpoint.get('num_classes', 40),
        img_size=checkpoint.get('img_size', 224),
        pretrained=False
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    return model, checkpoint


if __name__ == "__main__":
    # Test model creation
    print("Testing SwinPAR model creation...")
    model = create_swinpar_model()
    summary = get_model_summary(model)
    print("\nModel created successfully!")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output range: {output.min().item():.4f} - {output.max().item():.4f}")
