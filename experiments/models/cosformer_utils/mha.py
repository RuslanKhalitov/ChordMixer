import torch
import torch.nn as nn
import torch.nn.functional as F


class MHA(nn.Module):
    """O(n * d^2) Multi-Head Attention.

    If 'kernel' == 'relu' and 'use_cos' == True, the model from
    'COSFORMER : RETHINKING SOFTMAX IN ATTENTION' (https://openreview.net/pdf?id=Bl8CQrx2Up4)
    is constructed. If 'kernel' == 'elu' and 'use_cos' == False, this class implements the 
    model from: 'Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention'
    (https://arxiv.org/pdf/2006.16236.pdf).

    Attributes:
      d_model: (int) The full dimension of the hidden activations.
      n_heads: (int) Number of attention heads calculated in parallel.
      d_head: (int) Dimension of each head. d_model = n_heads * d_head.
      denom_eps: (float) Small positive constant that is added to the denominator of the
        linear self attention for stabilization. See self.linear_attention().
      kernel: (func) Kernel function used to approximate softmax. Either F.relu() or
        F.elu() + 1.
      attention_func: (func) Function used to compute linear attention. Either 
        self.linear_attention (without cos reweigting) or self.cos_linear_attention 
        (with cos reweighting).
      w_qkv: (nn.Linear) Used to calculate Q, K, V all at the same time.
      w_o: (nn.Linear) Applied to the output after self attention.
      dropout: (nn.Dropout) Applied after w_o.
    """

    def __init__(self, d_model, n_heads, use_cos, kernel, dropout, denom_eps, bias):
        """Initialized a MHA Module.

        Args:
          d_model: (int) The full dimension of the hidden activations.
          n_heads: (int) Number of attention heads calculated in parallel.
          use_cos: (bool) If True, the cos reweighting mechanism from 
            https://openreview.net/pdf?id=Bl8CQrx2Up4 is implemented and positional
            embeddings are not used. If false, sinusoidal positional embeddings are used.
          kernel: (str) If 'relu' is given, softmax is approximated with F.relu(). If 'elu'
            is given, F.elu() + 1 is used.
          dropout: (float) Dropout rate.
          denom_eps: (float) Small positive constant that is added to the denominator of the
            linear self attention for stabilization. See self.linear_attention().
          bias: (bool) Whether to add bias to all linear layers.
        """
        super(MHA, self).__init__()
        assert d_model % n_heads == 0, 'd_model must be a multiple of n_heads'
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = self.d_model // self.n_heads
        self.denom_eps = denom_eps

        # Probably should experiment with more different kernels.
        if kernel == 'relu':
            self.kernel = self.relu_kernel
        elif kernel == 'elu':
            self.kernel = self.elu_kernel
        else:
            raise NotImplementedError(
                "The only options for 'kernel' are 'relu and 'elu'.")

        if use_cos:
            self.attention_func = self.cos_linear_attention
        else:
            self.attention_func = self.linear_attention

        self.w_qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def apply_mask(self, x, mask):
        """Zeroes out elements specified by the attention mask.

        Args:
          x: (torch.Tensor of float32)[batch_size, seq_len, d_model] Tensor to be masked.
          mask: (torch.Tensor of bool)[batch_size, seq_len, 1] or None: True for elements that
            will be replaced with zero, False for the ones that will remain unchanged. If None,
            the function will return x unchanged.

        Returns:
          x: (torch.Tensor of float32)[batch_size, seq_len, d_model] Tensor after masking.
        """
        if not mask is None:
            x = x.masked_fill(mask, 0)

        return x

    def split_heads(self, x):
        """Splits the last dimension of a tensor d_model into [n_heads, d_head].

        Args:
          x: (torch.Tensor of float32)[batch_size, seq_len, d_model]

        Returns:
          x: (torch.Tensor of float32)[batch_size, seq_len, n_heads, d_head]    
        """
        batch_size, seq_len = x.shape[:2]
        # x -> [batch_size, seq_len, d_model]
        x = x.view(batch_size, seq_len, self.n_heads, self.d_head)
        # x -> [batch_size, seq_len, n_heads, d_head]

        return x

    def merge_heads(self, x):
        """Merges the 2 last dimensions of a tensor [n_heads, d_head] into d_model.

        Args:
          x: (torch.Tensor of float32)[batch_size, seq_len, n_heads, d_head] 

        Returns:
          x: (torch.Tensor of float32)[batch_size, seq_len, d_model]

        """
        batch_size, seq_len = x.shape[:2]
        # x -> [batch_size, seq_len, n_heads, d_head]
        x = x.view(batch_size, seq_len, self.d_model).contiguous()
        # x -> [batch_size, seq_len, d_model]

        return x

    def elu_kernel(self, x):
        """Kernel proposed in https://arxiv.org/pdf/2006.16236.pdf"""
        return F.elu(x) + 1

    def relu_kernel(self, x):
        """Kernel proposed in https://openreview.net/pdf?id=Bl8CQrx2Up4"""
        return F.relu(x)

    def linear_attention(self, q, k, v, weights=None):
        """Implements linear attention as proposed in https://arxiv.org/pdf/2006.16236.pdf

        Translated from tensorflow to pytorch based on:
        https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/kernel_attention.py

        Args:
          q, k, v: (torch.Tensor of float32)[batch_size, seq_len, n_heads, d_head] The query, key 
            and value tensors. The kernel function must be already applied to q and k. The
            attention mask must be already applied to k.
          weights: None. Unused.

        Retruns:
          output: (torch.Tensor of float32)[batch_size, seq_len, n_heads, d_head] The result of
            linear self attention.

        """
        # q, k, v -> [batch_size, seq_len, n_heads, d_head]
        kv = torch.einsum('bsnx,bsnz->bnxz', k, v)
        # kv -> [batch_size, n_heads, d_head, d_head]
        # add dropout here
        denominator = 1.0 / (torch.einsum('bsnd,bnd->bsn',
                                          q, k.sum(axis=1)) + self.denom_eps)
        # denominator -> [batch_size, seq_len, n_heads]

        output = torch.einsum('bsnx,bnxz,bsn->bsnz', q,
                              kv, denominator).contiguous()
        # output -> [batch_size, seq_len, n_heads, d_head]

        return output

    def cos_linear_attention(self, q, k, v, weights):
        """Implements linear attention with cos reweighting as in https://openreview.net/pdf?id=Bl8CQrx2Up4.

        Args:
          q, k, v: (torch.Tensor of float32)[batch_size, seq_len, n_heads, d_head] The query, key 
            and value tensors. The kernel function must be already applied to q and k. The
            attention mask must be already applied to k.
          weights: (tuple of (torch.Tensor of float32)[batch_size, seq_len])
            weights = (cos, sin), where cos[i, j] = cos(pi * i / 2 / M[i]) whre M[i] is the length
            of the i-th sample in the batch. Similarly for sin.

        Retruns:
          output: (torch.Tensor of float32)[batch_size, seq_len, n_heads, d_head] The result of
            linear self attention.
        """
        # q, k, v -> [batch_size, seq_len, n_heads, d_head]
        cos, sin = weights
        # cos, sin -> [batch_size, seq_len]
        q_cos = torch.einsum('bsnd,bs->bsnd', q, cos)
        q_sin = torch.einsum('bsnd,bs->bsnd', q, sin)
        k_cos = torch.einsum('bsnd,bs->bsnd', k, cos)
        k_sin = torch.einsum('bsnd,bs->bsnd', k, sin)
        # q_cos, q_sin, k_cos, k_sin -> [batch_size, seq_len, n_heads, d_head]

        kv_cos = torch.einsum('bsnx,bsnz->bnxz', k_cos, v)
        # kv_cos -> [batch_size, n_heads, d_head, d_head]
        qkv_cos = torch.einsum('bsnx,bnxz->bsnz', q_cos, kv_cos)
        # qkv_cos -> [batch_size, seq_len, n_heads, d_head]

        kv_sin = torch.einsum('bsnx,bsnz->bnxz', k_sin, v)
        # kv_sin -> [batch_size, n_heads, d_head, d_head]
        qkv_sin = torch.einsum('bsnx,bnxz->bsnz', q_sin, kv_sin)
        # qkv_sin -> [batch_size, seq_len, n_heads, d_head]

        # denominator
        denominator = 1.0 / (torch.einsum('bsnd,bnd->bsn', q_cos, k_cos.sum(axis=1))
                             + torch.einsum('bsnd,bnd->bsn',
                                            q_sin, k_sin.sum(axis=1))
                             + self.denom_eps)
        # denominator -> [batch_size, seq_len, n_heads]

        output = torch.einsum('bsnz,bsn->bsnz', qkv_cos +
                              qkv_sin, denominator).contiguous()
        # output -> [batch_size, seq_len, n_heads, d_head]

        return output

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
        # mask -> [batch_size, seq_len, 1] or None
        q, k, v = torch.chunk(self.w_qkv(x), 3, -1)
        # q, k, v -> [batch_size, seq_len, d_model]

        # A note about padding & masking in linear kernel attention:
        # In the f(Q) * (f(K^T) * V) attention, f(Q) is mutiplied by a dxd matrix.
        # Therefore padded elements must be removed from (f(K^T) * V).
        # This can be done by replacing padded elements (in the seq_len) dimension
        # of either f(K^T) or V). However, as seen in linear_attention, K is
        # summed in the seq_len dimension in the denominator, which means that
        # K must be masked.

        q = self.split_heads(self.kernel(q))
        k = self.split_heads(self.apply_mask(self.kernel(k), mask))
        v = self.split_heads(v)
        # q, k, v -> [batch_size, seq_len, n_heads, d_head]

        x = self.attention_func(q, k, v, weights)
        # x -> [batch_size, seq_len, n_heads, d_head]
        x = self.merge_heads(x)
        x = self.dropout(self.w_o(x))
        # x -> [batch_size, seq_len, d_model]

        return x
