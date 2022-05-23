import torch
import torch.nn as nn


class Positional_embeddings(nn.Module):
    """Sinusoidal Position Embeddings

    Stolen from:
    https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py

    Attributes:
      emb: (torch.Tensor of float32)[max_len, d_model] Matrix of sinusoidal embeddings.
    """

    def __init__(self, d_model, max_len):
        """Initializes a Positional_embeddings Module.

        Args:
          d_model: (int) Dimension of each embeddding vector.
          max_len: (int) Maximum allowed length for an input sequence.
        """
        super(Positional_embeddings, self).__init__()
        inv_freq = 1. / \
            (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        position = torch.arange(0, max_len, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer('emb', emb)

    def forward(self, x):
        """Implements forward pass.

        Args:
          x: (torch.Tensor of float32)[batch_size, seq_len, d_model] Input batch.

        Returns:
          (torch.Tensor of float32)[batch_size, seq_len, d_model] Positional embeddings
            for the first seq_len positions.
        """
        return self.emb[None, :x.shape[1], :].to(x).detach()
