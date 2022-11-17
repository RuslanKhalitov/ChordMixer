# Fork from
# https://github.com/davidsvy/cosformer-pytorch

import math
from .mha_block import MHA_block, MHA_block_rezero
from utils import Positional_embeddings
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_transformer(nn.Module):
    """O(n * d^2) Bi-directional Transformer that approximates softmax with kernels.

    If 'kernel' == 'relu' and 'use_cos' == True, the model from
    'COSFORMER : RETHINKING SOFTMAX IN ATTENTION' (https://openreview.net/pdf?id=Bl8CQrx2Up4)
    is constructed. If 'kernel' == 'elu' and 'use_cos' == False, this class implements the 
    model from: 'Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention'
    (https://arxiv.org/pdf/2006.16236.pdf).

    Attributes:
      use_cos: (bool) If True, the cos reweighting mechanism from 
        https://openreview.net/pdf?id=Bl8CQrx2Up4 is implemented and positional
        embeddings are not used. If False, sinusoidal positional embeddings are used.
      emb_in: (nn.Embedding) Input Embeddings.
      emb_pos: (model.utils.Positional_embeddings) Sinusoidal Position Embeddings.
      mha_blocks: (nn.ModuleList of nn.TransformerEncoderLayer) MHA blocks.
    """

    def __init__(self, use_cos, kernel, d_model, n_heads, n_layers, problem,
                 n_emb, ffn_ratio=4, rezero=True, ln_eps=1e-5, denom_eps=1e-5,
                 bias=False, dropout=0.2, max_len=1024, xavier=True):
        """Initializes a Kernel_transformer Module.

        Args:
          use_cos: (bool) If True, the cosine reweighting mechanism from 
            https://openreview.net/pdf?id=Bl8CQrx2Up4 is implemented and positional
            embeddings are not used. If False, sinusoidal positional embeddings are used.
          kernel: (str) If 'relu' is given, softmax is approximated with F.relu(). If 'elu'
            is given, F.elu() + 1 is used.
          d_model: (int) The full dimension of the hidden activations.
          n_heads: (int) Number of attention heads calculated in parallel.
          n_layers: (int) Number of Transformer Encoder Layers.
          n_emb: (int) Number of embedding tokens.
          ffn_ratio: (int) The dimension of the hidden activations in FFN is ffn_ratio * d_model.
            ln_eps: Not used.
          rezero: (bool) If True, the model utilizes the ReZero architecture from 
            https://arxiv.org/pdf/2003.04887.pdf . If False, the Pre-LN architecture from 
            https://arxiv.org/pdf/2002.04745.pdf is used.
          ln_eps: (float) A value added to the denominator of nn.LayerNorm for numerical stability.
            Ignored if rezero == True.
          denom_eps: (float) Small positive constant that is added to the denominator of the
            linear self attention for stabilization.
          bias: (bool) Whether to add bias to all linear layers.
          dropout: (float) Dropout rate.
          max_len: (int) Maximum allowed length for an input sequence.
          xavier: (bool) Whether to initialize all Linear layers with init.xavier_uniform_.
        """
        super(Kernel_transformer, self).__init__()
        self.use_cos = use_cos
        self.problem = problem
        if self.problem == 'genbank':
          self.emb_in = nn.Embedding(n_emb, d_model)
        elif self.problem == 'adding':
          self.emb_in = nn.Linear(2, d_model)
        self.emb_pos = Positional_embeddings(d_model, max_len)
        self.max_len = max_len

        Block_class = MHA_block_rezero if rezero else MHA_block
        self.mha_blocks = nn.ModuleList([])
        for _ in range(n_layers):
            block = Block_class(
                d_model=d_model,
                n_heads=n_heads,
                use_cos=use_cos,
                kernel=kernel,
                dropout=dropout,
                ffn_ratio=ffn_ratio,
                ln_eps=ln_eps,
                denom_eps=denom_eps,
                bias=bias
            )
            self.mha_blocks.append(block)

        if xavier:
            self.init_xavier_uniform()

    def init_xavier_uniform(self):
        """Initializes all Linear layers with init.xavier_uniform_.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def get_mask(self, lengths, max_len=None):
        """Creates Attention Mask.

        Args:
          lengths: (torch.Tensor of long)[batch_size]. Length of each input sequence.
          max_len: (int or None) Maximum length inside the batch. If None, assigns
            max_len = lengths.max()

        Returns:
          mask: (torch.Tensor of long)[batch_size, max_len]. Attention Mask where
            padded elements are 0 and valid ones are 1.s
        """
        # lens -> [batch_size]
        if max_len is None:
            max_len = lengths.max()
        mask = torch.arange(max_len)[None, :].to(lengths) < lengths[:, None]
        # mask -> [batch_size, max_len]

        return mask

    def get_cos_weights(self, lengths, max_len=None):
        """Returns cosine weights.

        Used for reweighting as described in https://openreview.net/pdf?id=Bl8CQrx2Up4.

        Args:
          lengths: (torch.Tensor of long)[batch_size]. Length of each input sequence.
          max_len: (int or None) Maximum length inside the batch. If None, assigns
            max_len = lengths.max()

        Returns:
          (cos, sin): (tuple of torch.Tensor of float32)[batch_size, seq_len]).
            cos[i, j] = cos(pi * i / 2 / M[i]) whre M[i] is the length
            of the i-th sample in the batch. Similarly for sin.
        """
        # lengths -> [batch_size]
        if max_len is None:
            max_len = lengths.max()
        # For each sample x in the batch, calculate M(x) = len(x)
        M = lengths
        # M -> [batch_size]
        idxs = math.pi / 2 * torch.arange(max_len).to(lengths)
        # idxs -> [max_len]
        idxs = torch.outer(1.0 / M, idxs)  # [..., None, None]
        # idxs -> [batch_size, max_len]

        cos = torch.cos(idxs).detach()
        sin = torch.sin(idxs).detach()
        # cos, sin -> [batch_size, max_len]

        return cos, sin

    def forward(self, input_ids, attention_mask=None, lengths=None):
        """Implements forward pass.

        Although some arguments are optional, for correct behavior, all args must
        be provided.

        Args:
          input_ids: torch.Tensor of long [batch_size, seq_len]. Indices of tokens.
          labels: For binary classification, torch.Tensor of float [batch_size]. 
            Labels for classification or bert pre-training.
          attention_mask: torch.Tensor of long or bool [batch_size, seq_len]. Has 0 for 
            elemsnts that will be masked and 1 for those that will remain unchanges. 
            Compatible with Huggigface pipeline. If None is given, masking will be ignored.
          lengths: torch.Tensor of long [batch_size]. Lengths of the unpadded sequences.
            Can be None.

        Retruns:
          x: (torch.Tensor of float32)[batch_size, seq_len, d_head]
        """
        # input_ids -> [batch_size, seq_len]
        # labels -> [batch_size]
        # attention_mask -> [batch_size, max_len] or None
        # lengths -> [batch_size] or None
        # weights ->  (tuple 2 X [batch_size, seq_len]) or None

        if lengths is None:
            lengths = torch.full(
                [input_ids.shape[0]], input_ids.shape[0], device=input_ids.device)

        input_ids = input_ids[:, :lengths.max()]

        if not attention_mask is None:
            attention_mask = attention_mask[:, :lengths.max()]
            attention_mask = torch.logical_not(
                attention_mask[..., None].bool())
        # attention_mask -> [batch_size, max_len, 1] or None

        if self.use_cos:
            cos_weights = self.get_cos_weights(lengths, lengths.max())
        else:
            cos_weights = None

        #input_ids = input_ids.to(dtype = torch.long)

        x = self.emb_in(input_ids)
        if not self.use_cos:
            x += self.emb_pos(x)


        for block in self.mha_blocks:
            x = block(x, attention_mask, cos_weights)

        return x
