

## Parameters

- `image_size`: int.  
Image size. If you have rectangular images, make sure your image size is the maximum of the width and height
- `patch_size`: int.  
Number of patches. `image_size` must be divisible by `patch_size`.  
The number of patches is: ` n = (image_size // patch_size) ** 2` and `n` **must be greater than 16**.
- `num_classes`: int.  
Number of classes to classify.
- `dim`: int.  
Last dimension of output tensor after linear transformation `nn.Linear(..., dim)`.
- `depth`: int.  
Number of Transformer blocks.
- `heads`: int.  
Number of heads in Multi-head Attention layer. 
- `mlp_dim`: int.  
Dimension of the MLP (FeedForward) layer. 
- `channels`: int, default `3`.  
Number of image's channels. 
- `dropout`: float between `[0, 1]`, default `0.`.  
Dropout rate. 
- `emb_dropout`: float between `[0, 1]`, default `0`.  
Embedding dropout rate.
- `pool`: string, either `cls` token pooling or `mean` pooling


## Two Stream ViT

<img src="./images/cross_vit.png" width="400px"></img>

<a href="https://arxiv.org/abs/2103.14899">This paper</a> proposes to have two vision transformers processing the image at different scales, cross attending to one every so often. They show improvements on top of the base vision transformer.

```python
import torch
from vit_pytorch.cross_vit import CrossViT

v = CrossViT(
    image_size = 256,
    num_classes = 1000,
    depth = 4,               # number of multi-scale encoding blocks
    sm_dim = 192,            # high res dimension
    sm_patch_size = 16,      # high res patch size (should be smaller than lg_patch_size)
    sm_enc_depth = 2,        # high res depth
    sm_enc_heads = 8,        # high res heads
    sm_enc_mlp_dim = 2048,   # high res feedforward dimension
    lg_dim = 384,            # low res dimension
    lg_patch_size = 64,      # low res patch size
    lg_enc_depth = 3,        # low res depth
    lg_enc_heads = 8,        # low res heads
    lg_enc_mlp_dim = 2048,   # low res feedforward dimensions
    cross_attn_depth = 2,    # cross attention rounds
    cross_attn_heads = 8,    # cross attention heads
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(1, 3, 256, 256)

pred = v(img) # (1, 1000)
```









## Accessing Attention

If you would like to visualize the attention weights (post-softmax) for your research, just follow the procedure below

```python
import torch
from vit_pytorch.vit import ViT

v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

# import Recorder and wrap the ViT

from vit_pytorch.recorder import Recorder
v = Recorder(v)

# forward pass now returns predictions and the attention maps

img = torch.randn(1, 3, 256, 256)
preds, attns = v(img)

# there is one extra patch due to the CLS token

attns # (1, 6, 16, 65, 65) - (batch x layers x heads x patch x patch)
```

to cleanup the class and the hooks once you have collected enough data

```python
v = v.eject()  # wrapper is discarded and original ViT instance is returned
```

## Accessing Embeddings

You can similarly access the embeddings with the `Extractor` wrapper

```python
import torch
from vit_pytorch.vit import ViT

v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

# import Recorder and wrap the ViT

from vit_pytorch.extractor import Extractor
v = Extractor(v)

# forward pass now returns predictions and the attention maps

img = torch.randn(1, 3, 256, 256)
logits, embeddings = v(img)

# there is one extra token due to the CLS token

embeddings # (1, 65, 1024) - (batch x patches x model dim)
```

Or say for `CrossViT`, which has a multi-scale encoder that outputs two sets of embeddings for 'large' and 'small' scales

```python
import torch
from vit_pytorch.cross_vit import CrossViT

v = CrossViT(
    image_size = 256,
    num_classes = 1000,
    depth = 4,
    sm_dim = 192,
    sm_patch_size = 16,
    sm_enc_depth = 2,
    sm_enc_heads = 8,
    sm_enc_mlp_dim = 2048,
    lg_dim = 384,
    lg_patch_size = 64,
    lg_enc_depth = 3,
    lg_enc_heads = 8,
    lg_enc_mlp_dim = 2048,
    cross_attn_depth = 2,
    cross_attn_heads = 8,
    dropout = 0.1,
    emb_dropout = 0.1
)

# wrap the CrossViT

from vit_pytorch.extractor import Extractor
v = Extractor(v, layer_name = 'multi_scale_encoder') # take embedding coming from the output of multi-scale-encoder

# forward pass now returns predictions and the attention maps

img = torch.randn(1, 3, 256, 256)
logits, embeddings = v(img)

# there is one extra token due to the CLS token

embeddings # ((1, 257, 192), (1, 17, 384)) - (batch x patches x dimension) <- large and small scales respectively
```

## Research Ideas

### D2F Attention



```bash
$ pip install DenseVH-attention
```

```python
import torch
from vit_pytorch.efficient import ViT
from d2f_attention import DenseVH

d2f_transformer = DenseVH(
    dim = 512,
    depth = 12,
    heads = 8,
    num_landmarks = 256
)

v = ViT(
    dim = 512,
    image_size = 2048,
    patch_size = 32,
    num_classes = 1000,
    transformer = d2f_transformer
)

img = torch.randn(1, 3, 2048, 2048) # your high resolution picture
v(img) # (1, 1000)
```

### Combining with other Transformer improvements

If you would like to use some of the latest improvements for attention nets

ex.

```bash
$ pip install x-transformers
```

```python
import torch
from vit_pytorch.efficient import ViT
from x_transformers import Encoder

v = ViT(
    dim = 512,
    image_size = 224,
    patch_size = 16,
    num_classes = 1000,
    transformer = Encoder(
        dim = 512,                  # set to be the same as the wrapper
        depth = 12,
        heads = 8,
        ff_glu = True,              # ex. feed forward GLU variant https://arxiv.org/abs/2002.05202
        residual_attn = True        # ex. residual attention https://arxiv.org/abs/2012.11747
    )
)

img = torch.randn(1, 3, 224, 224)
v(img) # (1, 1000)
```

## Citations
```bibtex
@Article{Wang2022,
  Title                    = {D2F: discriminative dense fusion of appearance and motion modalities for end-to-end video classification},
  Author                   = {Wang, Lin and Wang, Xingfu and Hawbani, Ammar and Xiong, Yan and Zhang, Xu},
  Journal                  = {Multimedia Tools and Applications},
  Year                     = {2022},
  Pages                    = {--},

  Abstract                 = {Recently, two-stream networks with multi-modality inputs have shown to be of vital importance for state-of-the-art video understanding. Previous deep systems typically employ a late fusion strategy, however, despite its simplicity and effectiveness, the late strategy might experience insufficient fusion due to that it performs fusion across modalities only once and treats each modality equally without discrimination. In this paper, we propose a Discriminative Dense Fusion (D2F) network, addressing these limitations by densely inserting an attention-based fusion block at each layer. We experiment with two typical action classification benchmarks and three popular classification backbones, where our proposed module consistently outperforms state-of-the-art baselines by noticeable margins. Specifically, the two-stream VGG16, ResNet and I3D achieve accuracy of [93.5%, 69.2%], [94.6%, 70.5%], [94.1%, 72.3%] with D2F on [UCF101, HMDB51], respectively, with absolute gains of [5.5%, 9.8%], [5.13%, 9.91%], and [0.7%, 5.9%] compared with their late fusion counterparts. The qualitative performance also demonstrates that our model can learn more informative complementary representation.},
  ISSN                     = {1573-7721},
  Owner                    = {Administrator},
  Refid                    = {Wang2022},
  Timestamp                = {2022.01.12},
  Url                      = {https://doi.org/10.1007/s11042-021-11247-7}
}

```



