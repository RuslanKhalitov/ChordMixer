import numpy as np
import torch
from torch.utils.data import Dataset

class VariableLengthsMask(Dataset):
    """
    A custom dataset class for variable lengths sequences masking.

    Args:
        df (DataFrame): A DataFrame containing sequences, labels, and lengths.
        mask_ratio (float, optional): Ratio of elements to mask in a sequence. Default is 0.2.
        mask_value (int, optional): Value to use for masked elements. Default is -1.

    Returns:
        A tuple containing sequences, labels, masks, masked sequences, and lengths.
    """
    def __init__(self, df, mask_ratio=0.2, mask_value=-1):
        self.df = df[['sequence', 'label', 'len']]
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        
    def __getitem__(self, indexes):
        """
        Returns: tuple (sample, target)
        """
        df = self.df.iloc[indexes, :]
        X = df['sequence'].to_list() #[array, array, array, array]
        Y = df['label'].to_list()    #[1, 0, 1, 0]
        length = df['len'].to_list() #[33, 36, 40, 42]
        s = sum(length) #length of the long sequence
        
        #Mask a batch simultaneously
        X = np.concatenate(X) #[array+array+array+array]
        sample_pix =  np.random.randint(low=0, high=s, size=int(s * self.mask_ratio)) # samples indices to mask
        mask = np.ones_like(X)
        mask[sample_pix] = 0 # all chosen indices are zeroed
        masked = X.copy()
        masked[sample_pix] = self.mask_value
        
        
        X = torch.from_numpy(X) # all sequences
        mask = torch.from_numpy(mask)
        masked = torch.from_numpy(masked)
        Y = torch.tensor(Y)
        length = torch.tensor(length)
        
        return (X, Y, mask, masked, length)

    def __len__(self):
        return len(self.df)