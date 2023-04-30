import torch 
import torch.nn as nn

class RotateChordCircularPad(nn.Module):
    """
    A PyTorch module that performs a parameter-free rotation of tracks within token embeddings with circular padding.

    The rotation is performed jointly for all sequences in a batch and is based on powers of 2 (Chord protocol). 
    The input tensor is first padded circularly before applying the rotation.
    
    For correct rotation, the circular padding should always be more than the maximum rotation = log2(N_batch_max).
    3 * log2(N_batch_min) > N_batch_max + log2(N_batch_max). This is achieved by bucket sampling, that ensures all sequences are within the same log2 bucket. 
    
    The resulting tensor shape: (B, N_batch_max, n_channels)

    Args:
        track_size (int): The size of tracks to be rotated.
    """
    def __init__(self, track_size):
        super().__init__()
        self.track_size = track_size

    def forward(self, x, lengths):
        padded_len = 3 * min(lengths)
        max_len = max(lengths)

        xs = []
        for i, x_i in enumerate(torch.split(x, 1, dim=0)):
            x_i = x_i.squeeze()[:lengths[i], :]
            xs.append(torch.cat([x_i, x_i, x_i], dim=0)[:padded_len, :])
        x = torch.stack(xs)

        y = torch.split(
            tensor=x,
            split_size_or_sections=self.track_size,
            dim=-1
        )
        
        # Roll sequences in a batch jointly
        z = [y[0]]
        for i in range(1, len(y)):
            offset = - 2 ** (i - 1)
            z.append(torch.roll(y[i], shifts=offset, dims=1))

        # Remove the unnecessary tokens
        z = torch.cat(z, -1)[:, :max_len, :]

        return z