from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import misc
from torch_utils import persistence



@persistence.persistent_class
class ScalarEncoder1d(nn.Module):
    """
    1-dimensional Fourier Features encoder (i.e. encodes raw scalars)
    Assumes that scalars are in [0, 1]
    """
    def __init__(self, coord_dim: int, x_multiplier: float, const_emb_dim: int, use_raw: bool=False, **fourier_enc_kwargs):
        super().__init__()
        self.coord_dim = coord_dim
        self.const_emb_dim = const_emb_dim
        self.x_multiplier = x_multiplier
        self.use_raw = use_raw

        if self.const_emb_dim > 0 and self.x_multiplier > 0:
            self.const_embed = nn.Embedding(int(np.ceil(x_multiplier)) + 1, self.const_emb_dim)
        else:
            self.const_embed = None

        if self.x_multiplier > 0:
            self.fourier_encoder = FourierEncoder1d(coord_dim, max_x_value=x_multiplier, **fourier_enc_kwargs)
            self.fourier_dim = self.fourier_encoder.get_dim()
        else:
            self.fourier_encoder = None
            self.fourier_dim = 0

        self.raw_dim = 1 if self.use_raw else 0

    def get_dim(self) -> int:
        return self.coord_dim * (self.const_emb_dim + self.fourier_dim + self.raw_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Assumes that x is in [0, 1] range
        """
        misc.assert_shape(x, [None, self.coord_dim])
        batch_size, coord_dim = x.shape
        out = torch.empty(batch_size, self.coord_dim, 0, device=x.device, dtype=x.dtype) # [batch_size, coord_dim, 0]
        if self.use_raw:
            out = torch.cat([out, x.unsqueeze(2)], dim=2) # [batch_size, coord_dim, 1]
        if not self.fourier_encoder is None or not self.const_embed is None:
            # Convert from [0, 1] to the [0, `x_multiplier`] range
            x = x.float() * self.x_multiplier # [batch_size, coord_dim]
        if not self.fourier_encoder is None:
            fourier_embs = self.fourier_encoder(x) # [batch_size, coord_dim, fourier_dim]
            out = torch.cat([out, fourier_embs], dim=2) # [batch_size, coord_dim, raw_dim + fourier_dim]
        if not self.const_embed is None:
            const_embs = self.const_embed(x.round().long()) # [batch_size, coord_dim, const_emb_dim]
            out = torch.cat([out, const_embs], dim=2) # [batch_size, coord_dim, raw_dim + fourier_dim + const_emb_dim]
        out = out.view(batch_size, coord_dim * (self.raw_dim + self.const_emb_dim + self.fourier_dim)) # [batch_size, coord_dim * (raw_dim + const_emb_dim + fourier_dim)]
        return out

#----------------------------------------------------------------------------

@persistence.persistent_class
class FourierEncoder1d(nn.Module):
    def __init__(self,
            coord_dim: int,               # Number of scalars to encode for each sample
            max_x_value: float=100.0,       # Maximum scalar value (influences the amount of fourier features we use)
            transformer_pe: bool=False,     # Whether we should use positional embeddings from Transformer
            use_cos: bool=True,
            **construct_freqs_kwargs,
        ):
        super().__init__()
        assert coord_dim >= 1, f"Wrong coord_dim: {coord_dim}"
        self.coord_dim = coord_dim
        self.use_cos = use_cos
        if transformer_pe:
            d_model = 512
            fourier_coefs = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)) # [d_model]
        else:
            fourier_coefs = construct_log_spaced_freqs(max_x_value, **construct_freqs_kwargs)
        self.register_buffer('fourier_coefs', fourier_coefs) # [num_fourier_feats]
        self.fourier_dim = self.fourier_coefs.shape[0]

    def get_dim(self) -> int:
        return self.fourier_dim * (2 if self.use_cos else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2, f"Wrong shape: {x.shape}"
        assert x.shape[1] == self.coord_dim
        fourier_raw_embs = self.fourier_coefs.view(1, 1, self.fourier_dim) * x.float().unsqueeze(2) # [batch_size, coord_dim, fourier_dim]
        if self.use_cos:
            fourier_embs = torch.cat([fourier_raw_embs.sin(), fourier_raw_embs.cos()], dim=2) # [batch_size, coord_dim, 2 * fourier_dim]
        else:
            fourier_embs = fourier_raw_embs.sin() # [batch_size, coord_dim, fourier_dim]
        return fourier_embs

#----------------------------------------------------------------------------

def construct_log_spaced_freqs(max_t: int, skip_small_t_freqs: int=0, skip_large_t_freqs: int=0) -> Tuple[int, torch.Tensor]:
    time_resolution = 2 ** np.ceil(np.log2(max_t))
    num_fourier_feats = np.ceil(np.log2(time_resolution)).astype(int)
    powers = torch.tensor([2]).repeat(num_fourier_feats).pow(torch.arange(num_fourier_feats)) # [num_fourier_feats]
    powers = powers[skip_large_t_freqs:len(powers) - skip_small_t_freqs] # [num_fourier_feats]
    fourier_coefs = powers.float() * np.pi # [1, num_fourier_feats]

    return fourier_coefs / time_resolution