---
title: "Vision Transformers (a small summary)"
date: 2025-01-05
categories:
  - deep-learning
  - computer-vision
tags:
  - transformers
  - vision-transformers
  - machine-learning
  - computer-vision
toc: true
toc_label: "Sections"
toc_icon: "file-alt"
toc_sticky: true
---

Vision Transformers (ViTs) have revolutionized computer vision by adapting the transformer architecture, originally designed for natural language processing, to image understanding tasks. This post explores how they work and why they've become so successful.

## The Core Idea

The fundamental insight behind Vision Transformers is surprisingly simple: treat an image as a sequence of patches, similar to how text is treated as a sequence of words. This allows us to leverage the powerful self-attention mechanisms that made transformers so successful in NLP.

## How Vision Transformers Work

### 1. Image Patching

The first step is to divide an input image into fixed-size patches:

```python
# Divide image into 16x16 patches
patches = image.unfold(2, 16, 16).unfold(3, 16, 16)
patches = patches.reshape(batch_size, -1, 3 * 16 * 16)
```

For a 224x224 image with 16x16 patches, this creates 196 patches (14x14 grid), each containing 768 values (16×16×3 for RGB).

### 2. Linear Projection

Each patch is then linearly projected to create patch embeddings:

```python
# Project patches to embedding dimension
patch_embeddings = linear_projection(patches)  # [B, 196, 768]
```

### 3. Position Embeddings

Since transformers don't inherently understand spatial relationships, we add positional embeddings:

```python
# Add learnable position embeddings
position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
x = patch_embeddings + position_embeddings[:, 1:]
```

### 4. Classification Token

A special [CLS] token is prepended to the sequence, which will be used for classification:

```python
# Prepend classification token
cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
x = torch.cat([cls_token.expand(batch_size, -1, -1), x], dim=1)
```

### 5. Transformer Encoder

The sequence then passes through standard transformer encoder layers:

```python
# Apply transformer blocks
for block in transformer_blocks:
    x = block(x)  # Self-attention + MLP
```

### 6. Classification Head

Finally, the [CLS] token representation is used for classification:

```python
# Use CLS token for classification
cls_output = x[:, 0]
logits = classification_head(cls_output)
```

## Key Advantages of Vision Transformers

### 1. Global Receptive Field
Unlike CNNs that build up receptive fields gradually, ViTs have global context from the first layer through self-attention.

### 2. Scalability
ViTs scale remarkably well with data and model size, following similar scaling laws to language models.

### 3. Flexibility
The same architecture works across different image resolutions and can handle various computer vision tasks.

### 4. Transfer Learning
Pre-trained ViTs transfer exceptionally well to downstream tasks, often outperforming specialized CNN architectures.

## Common Challenges and Solutions

### 1. Data Requirements
**Challenge**: ViTs typically need large datasets for training from scratch.
**Solution**: Use pre-trained models or hybrid CNN-transformer architectures for smaller datasets.

### 2. Computational Cost
**Challenge**: Self-attention has quadratic complexity with sequence length.
**Solution**: Techniques like windowed attention, hierarchical transformers, or efficient attention mechanisms.

### 3. Inductive Bias
**Challenge**: ViTs lack CNN's built-in spatial inductive biases.
**Solution**: Add convolutional stems, use hybrid architectures, or incorporate spatial priors in position embeddings.

## Popular Vision Transformer Variants

1. **DeiT (Data-efficient Image Transformers)**: Introduces distillation techniques for training with less data
2. **Swin Transformer**: Uses hierarchical structure with shifted windows for efficiency
3. **BEiT**: Applies BERT-style pre-training to vision transformers
4. **CvT**: Incorporates convolutional layers into transformer blocks

## Practical Implementation Tips

1. **Start with Pre-trained Models**: Use models pre-trained on ImageNet-21k or larger datasets
2. **Careful Hyperparameter Tuning**: Pay attention to learning rates, warmup schedules, and weight decay
3. **Data Augmentation**: Strong augmentation is crucial, especially for smaller datasets
4. **Mixed Precision Training**: Use fp16 to reduce memory requirements and speed up training

## Code Example: Simple ViT Forward Pass

```python
import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=1000, 
                 dim=768, depth=12, heads=12, mlp_dim=3072):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2
        
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, 
                                     dim_feedforward=mlp_dim),
            num_layers=depth
        )
        
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)
    
    def forward(self, img):
        p = self.patch_size
        x = img.unfold(2, p, p).unfold(3, p, p)  # Extract patches
        x = x.contiguous().view(img.shape[0], -1, 3 * p * p)
        
        x = self.patch_to_embedding(x)
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        
        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)
```

## Future Directions

The field of vision transformers is rapidly evolving with research focusing on:

1. **Efficiency Improvements**: Reducing computational complexity while maintaining performance
2. **Multi-modal Learning**: Combining vision and language in unified architectures
3. **Self-supervised Learning**: Developing better pre-training objectives for vision transformers
4. **Architecture Search**: Automatically discovering optimal transformer architectures for vision tasks

## Conclusion

Vision Transformers represent a significant paradigm shift in computer vision, demonstrating that the self-attention mechanism can be as powerful for images as it is for text. While challenges remain, particularly around computational efficiency and data requirements, the flexibility and scalability of ViTs make them an increasingly attractive choice for many computer vision applications.

As the field continues to evolve, we can expect to see more innovations that address current limitations while pushing the boundaries of what's possible in computer vision.

---

*Have questions about Vision Transformers or want to discuss their applications? Feel free to reach out!*