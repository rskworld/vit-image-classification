"""
Vision Transformer (ViT) Model Implementation
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module implements the Vision Transformer architecture for image classification.
It includes patch embedding, positional encoding, and multi-head self-attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    """
    Patch Embedding Layer
    Splits image into patches and projects them to embedding dimension
    
    Author: RSK World
    Website: https://rskworld.in
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        # x: (batch_size, channels, height, width)
        x = self.projection(x)  # (batch_size, embed_dim, n_patches^0.5, n_patches^0.5)
        x = rearrange(x, 'b e h w -> b (h w) e')  # (batch_size, n_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention Mechanism
    
    Author: RSK World
    Website: https://rskworld.in
    """
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    """
    Feed Forward Network
    
    Author: RSK World
    Website: https://rskworld.in
    """
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Transformer Block with Multi-Head Attention and Feed Forward
    
    Author: RSK World
    Website: https://rskworld.in
    """
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) Model
    
    Complete ViT architecture for image classification.
    Implements patch-based embedding, positional encoding, and transformer blocks.
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_classes=1000,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.1
    ):
        super().__init__()
        
        self.patch_embedding = PatchEmbedding(image_size, patch_size, 3, dim)
        num_patches = self.patch_embedding.n_patches
        
        # Learnable class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        # Transformer blocks
        self.transformer = nn.Sequential(
            *[TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)]
        )
        
        # Classification head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
    
    def forward(self, img):
        # Patch embedding
        x = self.patch_embedding(img)  # (batch_size, n_patches, dim)
        b, n, _ = x.shape
        
        # Add class token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, n_patches + 1, dim)
        
        # Add positional embedding
        x += self.pos_embedding
        x = self.dropout(x)
        
        # Apply transformer blocks
        x = self.transformer(x)
        
        # Classification
        x = self.norm(x)
        cls_token_final = x[:, 0]  # Extract class token
        return self.head(cls_token_final)


def create_vit_model(config):
    """
    Create ViT model from configuration
    
    Author: RSK World
    Website: https://rskworld.in
    """
    model_config = config['model']
    return VisionTransformer(
        image_size=model_config['image_size'],
        patch_size=model_config['patch_size'],
        num_classes=model_config['num_classes'],
        dim=model_config['dim'],
        depth=model_config['depth'],
        heads=model_config['heads'],
        mlp_dim=model_config['mlp_dim'],
        dropout=model_config['dropout'],
        emb_dropout=model_config['emb_dropout']
    )


if __name__ == "__main__":
    # Test the model
    # Author: RSK World
    # Website: https://rskworld.in
    
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        num_classes=1000,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072
    )
    
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

