import pandas as pd
import torch
import os
from torch.utils.data import Dataset
from utils.func import standardise
from config import ROOT_DIR

class SpermWhaleDataset(Dataset):
    def __init__(self, anootations_file, files_dir, target_length, sources='all', channels='all'):
        """Initialise SpermWhaleDataset dataset.

        Args:
            anootations_file (_type_): _description_
            files_dir (_type_): _description_
            target_length (_type_): _description_
            sources (str, optional): _description_. Defaults to 'all'.
            channels (str, optional): _description_. Defaults to 'all'.
        """
        self._df_annotations = self._load_annotations(anootations_file, sources)

        self._files_dir = files_dir

        self._target_length = target_length

        self._channels = channels

        print(f"There are {len(self._df_annotations)} in the SpermWhaleDataset") 

    def _load_annotations(self, anootations_file, sources):
        """Load dataframe of files and labels

        Args:
            anootations_file (_type_): _description_
            sources (_type_): _description_

        Returns:
            _type_: _description_
        """
        df_annotations = pd.read_csv(os.path.join(ROOT_DIR, anootations_file))
        if sources!='all':
            df_annotations = df_annotations[df_annotations.Dataset.isin(sources)].reset_index(drop=True)

        return df_annotations
    
    def annotations(self):
        """Get annotations dataframe.

        Returns:
            _type_: _description_
        """
        return self._df_annotations

    def __len__(self):
        return len(self._df_annotations)
	
    def _load_item(self, fname):
        """Load sequence of extracted features.

        Args:
            fname (_type_): _description_

        Returns:
            _type_: _description_
        """
        t = torch.load(os.path.join(self._files_dir, fname))
        
        # cut and get selected columns
        if self._channels!='all':
            t = t[self._channels, :]

        t = t[:, :self._target_length]

        #standardise
        for i in range(t.shape[0]):
            t[i, :] = standardise(t[i, :])

        t = self._right_pad_if_necessary(t)
         
        return t
    
    def _right_pad_if_necessary(self, signal, dim=1):
        """Right pad signal if below specified length.

        Args:
            signal (_type_): _description_
            dim (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        length_signal = signal.shape[dim]
        if length_signal < self._target_length:
            num_missing_samples = self._target_length - length_signal
            last_dim_padding = (0, num_missing_samples) # padding if necessaryy
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal  

    def __getitem__(self, index):
        label = self._df_annotations.Label[index]
        sequence = self._load_item(self._df_annotations.FileName[index][:-3]+'pt')

        return sequence, int(label)
    

    


