from models.utils import Positional_embeddings
import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline_transformer(nn.Module):
    """A makeshift O(n^2 * d) BERT-style transformer used as a baseline. 

    Attributes
      emb_in: (nn.Embedding) Input Embeddings.
      emb_pos: (model.utils.Positional_embeddings) Sinusoidal Position Embeddings.
      mha_blocks: (nn.ModuleList of nn.TransformerEncoderLayer) MHA blocks.
    """

    def __init__(self, **kwargs):
        """Initializes a Baseline_transformer Module.

        Args:
          **kwargs: (dict)
        """
        super(Baseline_transformer, self).__init__()
        self.emb_in = nn.Embedding(kwargs['n_emb'], kwargs['d_model'])
        self.emb_pos = Positional_embeddings(
            kwargs['d_model'], kwargs['max_len'])

        self.mha_blocks = nn.ModuleList([])
        for _ in range(kwargs['n_layers']):
            block = nn.TransformerEncoderLayer(
                d_model=kwargs['d_model'],
                nhead=kwargs['n_heads'],
                dim_feedforward=kwargs['d_model'] * kwargs['ffn_ratio'],
                dropout=kwargs['dropout'],
                activation='gelu',
                layer_norm_eps=kwargs['ln_eps'],
                batch_first=True,
            )
            self.mha_blocks.append(block)

        if kwargs['xavier']:
            self.init_xavier_uniform()

    def init_xavier_uniform(self):
        """Initializes all Linear layers with init.xavier_uniform_.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask=None, lengths=None):
        """Implements forward pass.

        Args:
          input_ids: torch.Tensor of long [batch_size, seq_len]. Indices of tokens.
          labels: Labels for classification or bert pre-training.
          attention_mask: torch.Tensor of long [batch_size, seq_len]. Has 0 for elemsnts
            that will be masked and 1 for those that will remain unchanges. If None is
            given, masking will be ignored.
          lengths: torch.Tensor of long [batch_size]. Lengths of the unpadded sequences.
            Can be None.

        Returns:
          (tuple): loss, logits
        """
        # input_idxs -> [batch_size, seq_len]
        # labels -> [batch_size]
        # attention_mask -> [batch_size, max_len] or None
        # lengths -> [batch_size]

        if lengths is None:
            lengths = torch.full(
                [input_ids.shape[0]], input_ids.shape[0], device=input_ids.device)

        input_ids = input_ids[:, :lengths.max()]

        if not attention_mask is None:
            attention_mask = torch.logical_not(attention_mask.bool())
        # attention_mask -> [batch_size, max_len] or None

        x = self.emb_in(input_ids)
        x += self.emb_pos(x)

        for block in self.mha_blocks:
            x = block(x, src_key_padding_mask=attention_mask)

        return x
