from .mha import MHA
import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    """Feed Forward Neural Network used in the vanilla Transformer.

    Attributes:
      layers: (nn.Sequential) The layers of the Feed Forward Neural Network.
    """

    def __init__(self, d_model, ffn_ratio, dropout, bias):
        """Initializes a FFN Module.

        Args:
          d_model: (int) The input and output dimension of the module.
          ffn_ratio: (int) The dimension of the hidden activations is ffn_ratio * d_model.
          dropout: (float) Dropout rate.
          bias: (bool) Whether to add bias to all linear layers.
        """
        super(FFN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, ffn_ratio * d_model, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_ratio * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """Implements forward pass.

        Args: 
          x: (torch.Tensor of float32)[batch_size, seq_len, d_model] Input batch.

        Returns: 
          x: (torch.Tensor of float32)[batch_size, seq_len, d_model]
        """
        return self.layers(x)


class MHA_block(nn.Module):
    """Implements the Pre-LN Architecture as in https://arxiv.org/pdf/2002.04745.pdf

    Attributes:
      ln1, ln2: (nn.LayerNorm) Layer Normalization Modules.
      mha: (models.mha.MHA) Custom Self Attention Module.
      ffn: (models.mha_block.FFN) Custom Feed Forward Module.
    """

    def __init__(self, d_model, n_heads, use_cos, kernel, dropout,
                 ffn_ratio, ln_eps, denom_eps, bias):
        """Initializes a MHA_block Module.

        Args:
          d_model: (int) The full dimension of the hidden activations.
          n_heads: (int) Number of attention heads calculated in parallel.
          use_cos: (bool) If True, the cos reweighting mechanism from 
            https://openreview.net/pdf?id=Bl8CQrx2Up4 is implemented and positional
            embeddings are not used. If false, sinusoidal positional embeddings are used.
          kernel: (str) If 'relu' is given, softmax is approximated with F.relu(). If 'elu'
            is given, F.elu() + 1 is used.
          dropout: (float) Dropout rate.
          ffn_ratio: (int) The dimension of the hidden activations in FFN is ffn_ratio * d_model.
          ln_eps: (float) A value added to the denominator of nn.LayerNorm for numerical stability.
          denom_eps: (float) Small positive constant that is added to the denominator of the
            linear self attention for stabilization. See self.linear_attention().
          bias: (bool) Whether to add bias to all linear layers.
        """
        super(MHA_block, self).__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=ln_eps)
        self.ln2 = nn.LayerNorm(d_model, eps=ln_eps)
        self.mha = MHA(
            d_model, n_heads, use_cos, kernel, dropout, denom_eps, bias)
        self.ffn = FFN(d_model, ffn_ratio, dropout, bias)

    def forward(self, x, mask, weights):
        """Implements forward pass.

        Args:
          x: (torch.Tensor of float32)[batch_size, seq_len, d_model] Input batch.
          mask: (torch.Tensor of bool)[batch_size, seq_len, 1] Attention mask.
            True for elements that must be masked. If mask is None, masking is not 
            applied.
          weights: (tuple of (torch.Tensor of float32)[batch_size, seq_len, n_heads, d_head])
            weights = (cos, sin), where cos[i, j] = cos(pi * i / 2 / M[i]) whre M[i] 
            is the length of the i-th sample in the batch. Similarly for sin. If cos
            reweighting is not applied, weights = None. 

        Retruns:
          x: (torch.Tensor of float32)[batch_size, seq_len, d_head]
        """
        # x -> [batch_size, seq_len, d_model]
        fx = self.mha(self.ln1(x), mask, weights)
        x = x + fx

        fx = self.ffn(self.ln2(x))
        x = x + fx
        # x -> [batch_size, seq_len, d_model]

        return x


class MHA_block_rezero(nn.Module):
    """Implements the ReZero Architecture as in https://arxiv.org/pdf/2003.04887.pdf

    Attributes:
      mha: (models.mha.MHA) Custom Self Attention Module.
      ffn: (models.mha_block.FFN) Custom Feed Forward Module.
      alpha: (nn.Parameter) Scalar that is multiplied with the residual.
    """

    def __init__(self, d_model, n_heads, use_cos, kernel, dropout,
                 ffn_ratio, ln_eps, denom_eps, bias):
        """Initializes a MHA_block_rezero Module.

        Args:
          d_model: (int) The full dimension of the hidden activations.
          n_heads: (int) Number of attention heads calculated in parallel.
          use_cos: (bool) If True, the cos reweighting mechanism from 
            https://openreview.net/pdf?id=Bl8CQrx2Up4 is implemented and positional
            embeddings are not used. If false, sinusoidal positional embeddings are used.
          kernel: (str) If 'relu' is given, softmax is approximated with F.relu(). If 'elu'
            is given, F.elu() + 1 is used.
          dropout: (float) Dropout rate.
          ffn_ratio: (int) The dimension of the hidden activations in FFN is ffn_ratio * d_model.
          ln_eps: Not used.
          denom_eps: (float) Small positive constant that is added to the denominator of the
            linear self attention for stabilization. See self.linear_attention().
          bias: (bool) Whether to add bias to all linear layers.
        """
        super(MHA_block_rezero, self).__init__()
        self.mha = MHA(
            d_model, n_heads, use_cos, kernel, dropout, denom_eps, bias)
        self.ffn = FFN(d_model, ffn_ratio, dropout, bias)
        self.alpha = nn.Parameter(torch.Tensor([0]))

    def forward(self, x, mask, weights):
        """Implements forward pass.

        Args:
          x: (torch.Tensor of float32)[batch_size, seq_len, d_model] Input batch.
          mask: (torch.Tensor of bool)[batch_size, seq_len, 1] Attention mask.
            True for elements that must be masked. If mask is None, masking is not 
            applied.
          weights: (tuple of (torch.Tensor of float32)[batch_size, seq_len, n_heads, d_head])
            weights = (cos, sin), where cos[i, j] = cos(pi * i / 2 / M[i]) whre M[i] 
            is the length of the i-th sample in the batch. Similarly for sin. If cos
            reweighting is not applied, weights = None. 

        Retruns:
          x: (torch.Tensor of float32)[batch_size, seq_len, d_head]
        """
        # x -> [batch_size, seq_len, d_model]
        fx = self.alpha * self.mha(x, mask, weights)
        x = x + fx

        fx = self.alpha * self.ffn(x)
        x = x + fx
        # x -> [batch_size, seq_len, d_model]

        return x
